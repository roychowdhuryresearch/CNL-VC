import os
import shutil
folder = "/media/yipeng/data/movie/data_movie_analysis/LSTM_multi_2_KLD"
patient_dir = os.listdir(folder)
for i in patient_dir: 
    patient_folder = os.path.join(folder,i)
    if os.path.isdir(patient_folder): 
        kfold_dir = os.listdir(os.path.join(patient_folder))
        for k in kfold_dir:
            kfold_folder = os.path.join(patient_folder,k)
            if os.path.isdir(kfold_folder):
                dnk_y = os.path.join(kfold_folder,"dnk_as_yes")
                dnk_n = os.path.join(kfold_folder,"dnk_as_no")
                tp = os.path.join(kfold_folder,"fp_plot")
                fp = os.path.join(kfold_folder,"fn_plot")
                if os.path.exists(dnk_y):
                    shutil.rmtree(dnk_y)
                if os.path.exists(dnk_n):
                    shutil.rmtree(dnk_n)
                if os.path.exists(tp):
                    shutil.rmtree(tp)
                if os.path.exists(fp):
                    shutil.rmtree(fp)



            
