# coding=utf-8
# Implements several parameter-efficient supervised fine-tuning method for ChatGLM.
# This code is inspired by https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
import datasets

from utils import (
    DataCollatorForChatGLM,
    Seq2SeqTrainerForChatGLM,
    ComputeMetrics,
    LogCallback,
    load_pretrained,
    prepare_args,
    prepare_data,
    preprocess_data,
    preprocess_squad_data,
    get_logits_processor,
    plot_loss
)


def main():

    # Prepare pretrained model and dataset
    model_args, data_args, training_args, finetuning_args = prepare_args(stage="sft")
    file_path = "/home/algroup/fhb/ActiveLLM/ChatGLM-Efficient-Tuning-main/data/squad/squad_v2.py"
    data_files = {"train": "squad_train-v2.0.json", "dev": "squad_train-v2.0.json"}
    raw_dataset = datasets.load_dataset(file_path, data_files=data_files)
    print(raw_dataset)
    # training_set, dev_set = prepare_data(model_args, data_args)
    training_set = raw_dataset["train"]
    dev_set = raw_dataset["validation"]

    model, tokenizer = load_pretrained(model_args, finetuning_args, training_args.do_train, stage="sft")

    training_set = preprocess_squad_data(training_set, tokenizer, data_args, training_args, stage="qg")
    dev_set = preprocess_squad_data(dev_set, tokenizer, data_args, training_args, stage="qg")

    data_collator = DataCollatorForChatGLM(tokenizer, model, data_args.ignore_pad_token_for_loss, use_v2=model_args.use_v2)

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length if \
                training_args.generation_max_length is not None else data_args.max_target_length
    training_args.generation_num_beams = data_args.eval_num_beams if \
                data_args.eval_num_beams is not None else training_args.generation_num_beams

    # Split the dataset
    if training_args.do_train:
        trainer_kwargs = {"train_dataset": training_set, "eval_dataset": dev_set}
    else: # do_eval or do_predict
        trainer_kwargs = {"eval_dataset": dev_set}

    # Initialize our Trainer
    trainer = Seq2SeqTrainerForChatGLM(
        finetuning_args=finetuning_args,
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[LogCallback()],
        compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
        **trainer_kwargs
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = {
        "do_sample": True,
        "top_p": 0.7,
        "max_new_tokens": data_args.max_target_length + 1,
        "temperature": 0.95,
        "logits_processor": get_logits_processor()
    }

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        trainer.save_model()
        if trainer.is_world_process_zero() and model_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if training_args.predict_with_generate: # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset, metric_key_prefix="predict", **gen_kwargs)
        if training_args.predict_with_generate: # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results, tokenizer)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
