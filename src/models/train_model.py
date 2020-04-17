import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Embedding, GRU, Dropout, Dense, BatchNormalization
from sacred import Experiment
ex = Experiment()
from text_generator import Keras_Text_Generator


def create_model(gen):

    vocab_size = gen.vocab_size
    embedding_dim = 400

    gen.add_layer_to_model(Embedding, 
                        input_dim=vocab_size, 
                        output_dim=embedding_dim)
    gen.add_layer_to_model(GRU,
                        units=400,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform')
    gen.add_layer_to_model(BatchNormalization)
    gen.add_layer_to_model(Dropout,
                        rate=0.1)
    gen.add_layer_to_model(GRU,
                        units=300,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform')
    gen.add_layer_to_model(BatchNormalization)
    gen.add_layer_to_model(Dropout,
                        rate=0.1)
    gen.add_layer_to_model(Dense,
                        units=vocab_size)

    gen.compile_model()

    return gen


@ex.main
def main(raw_data_path=snakemake.input.rs, text=snakemake.input.ms):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Training model...')

    gen = Keras_Text_Generator()

    gen.load_and_create_dataset(text, from_pickle=raw_data_path)

    gen = create_model(gen)

    gen.fit_model(epochs=2)

    logger.info('Experiment complete!')

    return None


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    # project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    ex.run()