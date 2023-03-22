import os
import pandas as pd
from asag import train_eval

path_dir = 'data'
list_dir = os.listdir(path_dir)

list_pre_trained_model = ['indobenchmark/indobert-lite-base-p2']

for m in list_pre_trained_model:
    print(m)
    for idx, ele in enumerate(list_dir):
        df_raw = pd.read_excel(open(path_dir+'/'+ele, 'rb'),
                               sheet_name='Soal',
                               header=1,
                               index_col=0,
                               usecols='B:D')

        list_final = []

        for i in df_raw.itertuples():
            list_final.append(
                {
                    'soal': i[1],
                    'jawaban': i[2],
                    'nilai': 100,
                    'tipe': 'train'
                }
            )
            df_tmp = pd.read_excel(open(path_dir+'/'+ele, 'rb'),
                                        sheet_name='No.'+str(i.Index),
                                        header=1,
                                        index_col=0,
                                        usecols='B:N')
            df_tmp = df_tmp.dropna()
            for j in df_tmp.itertuples():
                list_final.append(
                    {
                        'soal': i[1],
                        'jawaban': j[2],
                        'nilai': j[12],
                        'tipe': 'test'
                    }
                )
        if idx == 0:
            df_final = pd.DataFrame(list_final)
        else:
            df_final.append(pd.DataFrame(list_final), ignore_index=True)

        print(' '.join(ele.rstrip('.xslx').split('_')))
        train_eval(df_final, m)
