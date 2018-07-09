from model.data_utils import CoNLLDataset
from model.config import Config
from model.allen_model import ElmoModel
from model.ner_learner import NERLearner


def main():
    # create instance of config
    config = Config()
    config.use_chars = False

    #build model
    model = ElmoModel(config)

    # create datasets
    dev = CoNLLDataset(config.filename_dev, None, #filename_dev
                         config.processing_tag, config.max_iter)
    train = CoNLLDataset(config.filename_train, None,
                         config.processing_tag, config.max_iter)


    learn = NERLearner(config, model)
    learn.fit(train, dev)


if __name__ == "__main__":
    main()

