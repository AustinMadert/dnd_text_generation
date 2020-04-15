# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import re


def load_csv_file_to_df(path):
    return pd.read_csv(path)


def handle_na_and_duplicates(df):

    df.fillna('None', inplace=True)

    df.drop_duplicates(inplace=True)

    df.drop([616, 683], inplace=True)

    df.reset_index(drop=True, inplace=True)

    return df


def clean_text(text):

    text = str(text).replace('.;;', ':')

    text = text.replace('||ADDENDUM||', '')

    text = text.replace(',,,,,,,,,,,','')

    return text.lower()


def remove_parens(text):
    return re.sub('\(.*?\)','', string=text)


def clean_dataframe(df):

    df['challenge'] = df['challenge'].map(remove_parens)
    df['strength'] = df['strength'].map(remove_parens)
    df['dexterity'] = df['dexterity'].map(remove_parens)
    df['constitution'] = df['constitution'].map(remove_parens)
    df['wisdom'] = df['wisdom'].map(remove_parens)
    df['intelligence'] = df['intelligence'].map(remove_parens)
    df['charisma'] = df['charisma'].map(remove_parens)
    df['short_desc'] = df['short_desc'].map(remove_parens)

    return df.applymap(clean_text)


def organize_columns(df):

    columns_order = ['challenge', 'strength', 'dexterity', 'constitution', 'wisdom', 
            'intelligence', 'charisma', 'short_desc',
            'ability_1', 'ability_2', 'ability_3', 'ability_4', 'ability_5', 'ability_6', 'ability_7',
            'ability_8', 'ability_9', 'ability_10', 'ability_11', 'ability_12', 'ability_13', 'ability_14',
            'ability_15', 'ability_16', 'ability_17', 'ability_18', 'ability_19', 'ability_20']

    df = df[columns_order]

    abbreviated_cols = ['CR', 'STR', 'DEX', 'CON', 'WIS', 'INT', 'CHA', 'SD', 
                        'ability_1', 'ability_2', 'ability_3', 'ability_4', 'ability_5', 'ability_6', 'ability_7',
            'ability_8', 'ability_9', 'ability_10', 'ability_11', 'ability_12', 'ability_13', 'ability_14',
            'ability_15', 'ability_16', 'ability_17', 'ability_18', 'ability_19', 'ability_20']

    df.columns = abbreviated_cols

    return df


def concat_abilities(df):

    df['A'] = (df['ability_1'].map(str) + " " +
                df['ability_2'].map(str) + " " +
                df['ability_3'].map(str) + " " +
                df['ability_4'].map(str) + " " +
                df['ability_5'].map(str) + " " +
                df['ability_6'].map(str) + " " +
                df['ability_7'].map(str) + " " +
                df['ability_8'].map(str) + " " +
                df['ability_9'].map(str) + " " +
                df['ability_10'].map(str) + " " +
                df['ability_11'].map(str) + " " +
                df['ability_12'].map(str) + " " +
                df['ability_13'].map(str) + " " +
                df['ability_14'].map(str) + " " +
                df['ability_15'].map(str) + " " +
                df['ability_16'].map(str) + " " +
                df['ability_17'].map(str) + " " +
                df['ability_18'].map(str) + " " +
                df['ability_19'].map(str) + " " +
                df['ability_20'].map(str))

    df['A'] = df['A'].apply(lambda x: x.strip('none'))

    df['ability_len'] = df['A'].apply(lambda x: len(x))

    return df


def concat_monsters(df, start_str='<start>', end_str='<end>'):

    monsters_list = []

    for row in range(df.shape[0]):

        monster = start_str

        for idx, col in enumerate(df.columns):

            monster += ("[" + col + "]" + str(df.iloc[row, idx]))

        monster += end_str

        monsters_list.append(monster)

    return ''.join(monsters_list)


def create_monsters_string(df, ability_ceil = 1500):

    df = df[df['ability_len'] < ability_ceil]

    stats_with_abilities_cols = ['CR', 'STR', 'DEX', 'CON', 'WIS', 'INT', 'CHA', 
                                'SD', 'A']

    df = df[stats_with_abilities_cols]

    return concat_monsters(df)


def create_text_output_file(text, outpath):

    with open(outpath, 'w') as f:

        f.write(text)

    return None


def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data...')

    df = load_csv_file_to_df(input_filepath)
    df = handle_na_and_duplicates(df)
    df = clean_dataframe(df)
    df = organize_columns(df)
    df = concat_abilities(df)
    out_str = create_monsters_string(df)
    create_text_output_file(out_str, output_filepath)

    logger.info('Output file created!')

    return None


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(snakemake.input[0], snakemake.output[0])
