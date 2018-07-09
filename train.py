from model.data_utils import CoNLLDataset
from model.config import Config
from model.glove_model import NERModel
from model.ner_learner import NERLearner


def main():
    # create instance of config
    config = Config()

    #build model
    model = NERModel(config)

    # create datasets
    dev = CoNLLDataset(config.filename_dev, config.processing_word, #filename_dev
                         config.processing_tag, config.max_iter)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)

    learn = NERLearner(config, model)
    learn.load_emb()
    learn.fit(train, dev)


    # Save model


    # test predictions
    # words = "Obama was born in hawaii"
    # words = words.split(" ")
    # pred = model.predict_words(words)
    # print(pred)

if __name__ == "__main__":
    main()

