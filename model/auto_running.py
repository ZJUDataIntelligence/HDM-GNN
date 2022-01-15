import os

# 每次跑实验前，需要手动建立config_folder这个文件夹和其中的所有配置文件，然后修改这里的config_folder和purpose_name
config_folder = '/home/zh/pycharm_projects/crime_prediction/config/full_model_T_B'
purpose_name = 'full_model_T_B'  # 不要有空格

output_path = '/home/zh/pycharm_projects/outputs/our_model/' + purpose_name + '/'
if not os.path.exists(output_path):
    os.mkdir(output_path)

config_files = os.listdir(config_folder)
num_cfs = len(config_files)
count = 0
for cf in config_files:
    one_cf_output_path = output_path + cf
    if not os.path.exists(one_cf_output_path):
        os.mkdir(one_cf_output_path)
    os.system("cp {} {}".format(config_folder + '/' + cf, one_cf_output_path + '/' + cf))
    if count == 0 or count == 1:
        os.system("nohup /usr/local/anaconda3/bin/python -u train.py --config \"{}\" --out \"{}\" > {} &"
                  .format(config_folder + '/' + cf, one_cf_output_path + '/', one_cf_output_path + '/output.txt'))
        print("nohup /usr/local/anaconda3/bin/python -u train.py --config \"{}\" --out \"{}\" > {} &".format(
            config_folder + '/' + cf, one_cf_output_path + '/', one_cf_output_path + '/output.txt'))
        count += 1
    else:
        os.system("/usr/local/anaconda3/bin/python -u train.py --config \"{}\" --out \"{}\" > {}"
                  .format(config_folder + '/' + cf, one_cf_output_path + '/', one_cf_output_path + '/output.txt'))
        print("/usr/local/anaconda3/bin/python -u train.py --config \"{}\" --out \"{}\" > {}"
              .format(config_folder + '/' + cf, one_cf_output_path + '/', one_cf_output_path + '/output.txt'))
        count = 0
# print("s\"d\"a")
