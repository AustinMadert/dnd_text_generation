# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pickle
import tensorflow as tf
from text_generator import Keras_Text_Generator


def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('building features...')

    gen = Keras_Text_Generator()

    data = gen.load_and_create_dataset(input_filepath, seq_length=300)

    with open(output_filepath, 'wb') as f:
        pickle.dump(data, f)


    # writer = tf.data.experimental.TFRecordWriter(output_filepath)
    # writer.write(gen.dataset)
    # with open(output_filepath, 'wb') as f:
    #     pickle.dump(gen.dataset, f)  

    logger.info('Features built!')

    return None


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    # project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main(snakemake.input[0], snakemake.output.rs)