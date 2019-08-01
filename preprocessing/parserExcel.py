import pandas as pd


def get_dataframe_from_excel(path, index=None, skip_from_head=None, out_col=None, col_name_row=-1):
    """
    Function that make DataFrame from excel file
    :param path: str , path of the file to read
    :param index: int , index column used as index in the dataframe
    :param skip_from_head: int , number of rows to skip from the head of the csv
    :param out_col: list , list of columns that identify the target
    :param col_name_row: int , row index of the columns' names.
                        col_name_row == -1 means that no labels exist for the columns, so they will be created
    :return: DataFrame from excel file
    """

    df = pd.read_csv(path, index_col=index, skiprows=skip_from_head, header=col_name_row)

    if col_name_row < 0:
        col_name = ["x_" + str(i) for i in range(len(df.columns))]
        if out_col is not None:
            for count, i in enumerate(out_col):
                col_name[i] = "y_" + str(count)
        df.columns = col_name

    return df


def save_dataframe_into_excel(df, path):
    """
    Function that save DataFrame into excel file
    :param df: DataFrame
    :param path: str
    """

    # Write DataFrame to a excel sheet
    df.to_csv(path)


def dataframe2list(df):
    """
    Transform from dataframe into list
    :param df: DataFrame
    """
    return df.values.tolist()


def one_hot_encoding(df, cols):
    return pd.get_dummies(df, columns=cols) 
