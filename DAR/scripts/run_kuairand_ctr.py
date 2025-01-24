import sys
sys.path.append("..")
import torch
import pandas as pd
from tqdm import tqdm
from scenario_wise_rec.basic.features import DenseFeature, SparseFeature
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scenario_wise_rec.trainers import CTRTrainer
from scenario_wise_rec.utils.data import DataGenerator
from scenario_wise_rec.models.multi_domain import DAR

def get_kuairand_data_multidomain(data_path="/root/Test/KDD/Scenario-Wise-Rec_1/scripts/data/kuairand/"):
    data = pd.read_csv(data_path+"/kuairand_sample.csv")
    data = data[data["tab"].apply(lambda x: x in [1, 0, 2])]
    data.reset_index(drop=True, inplace=True)
    data.rename(columns={'tab': "domain_indicator"}, inplace=True)

    #这里因为sample的csv力domian都是同一个所以改下数据。。
    # n = len(data)
    # first_third_index = n // 3
    # second_third_index = 2 * (n // 3)
    # data.loc[:first_third_index - 1, 'domain_indicator'] = 2
    # data.loc[first_third_index:second_third_index - 1, 'domain_indicator'] = 0

    domain_num = data.domain_indicator.nunique()
    col_names = data.columns.to_list()
    dense_features = ["follow_user_num", "fans_user_num", "friend_user_num", "register_days"]
    useless_features = ["play_time_ms", "duration_ms", "profile_stay_time", "comment_stay_time"]
    scenario_features = ["domain_indicator"]
    sparse_features = [col for col in col_names if col not in dense_features and
                       col not in useless_features and col not in ['is_click','domain_indicator']]
    # target = "is_click"
    for feature in dense_features:
        data[feature] = data[feature].apply(lambda x: convert_numeric(x))
    if dense_features:
        sca = MinMaxScaler()  # scaler dense feature
        data[dense_features] = sca.fit_transform(data[dense_features])
    for feature in useless_features:
        del data[feature]
    for feature in scenario_features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature])
    for feature in tqdm(sparse_features):  # encode sparse feature
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature])

    dense_feas = [DenseFeature(feature_name) for feature_name in dense_features]
    sparse_feas = [SparseFeature(feature_name, vocab_size=data[feature_name].nunique(), embed_dim=16) for feature_name
                   in sparse_features]
    scenario_feas = [SparseFeature(col, vocab_size=data[col].max() + 1, embed_dim=16) for col in scenario_features]
    y=data["is_click"]
    del data["is_click"]
    return dense_feas, sparse_feas, scenario_feas, data, y, domain_num

def convert_numeric(val):
    """
    Forced conversion
    """
    return int(val)

def main(dataset_path, model_name, epoch, learning_rate, batch_size, weight_decay, device, save_dir, seed, ckpt_path):
    torch.manual_seed(seed)
    dataset_name = "Kuairand"
    dense_feas, sparse_feas, scenario_feas, x, y, domain_num = get_kuairand_data_multidomain(dataset_path)
    dg = DataGenerator(x, y)
    train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(split_ratio=[0.8, 0.1], batch_size=batch_size)
    model = DAR(
        features=dense_feas + sparse_feas,
        domain_num=domain_num,
        expert_num=domain_num,
        expert_params={'dims': [32]}, tower_params={'dims': [16]},
        args=args
    )
    #------------------------------training-----------------------------------
    if ckpt_path is None:
        ctr_trainer = CTRTrainer(model, dataset_name, optimizer_params={"lr": learning_rate, "weight_decay": weight_decay}, scheduler_params={"step_size": 4,"gamma": 0.75}, n_epoch=epoch, earlystop_patience=4, device=device, model_path=save_dir)
        ctr_trainer.fit(train_dataloader, val_dataloader)
        domain_logloss,domain_auc,logloss,auc = ctr_trainer.evaluate_multi_domain_loss(ctr_trainer.model, test_dataloader,domain_num)
        print(f'test auc: {auc} | test logloss: {logloss}')
        for d in range(domain_num):
            print(f'test domain {d} auc: {domain_auc[d]} | test domain {d} logloss: {domain_logloss[d]}')
        import csv
        with open(model_name+"_"+dataset_name+"_"+str(seed)+'.csv', "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['model', 'seed', 'auc', 'log', 'auc0', 'log0',
                                'auc1', 'log1', 'auc2', 'log2'])
            writer.writerow([model_name, str(seed), auc, logloss,
                                domain_auc[0], domain_logloss[0],
                                domain_auc[1], domain_logloss[1],
                                domain_auc[2], domain_logloss[2]]) 
    else:
        pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="/root/Test/KDD/Scenario-Wise-Rec_1/scripts/data/kuairand/")
    parser.add_argument('--model_name', default='DPRP')
    parser.add_argument('--epoch', type=int, default=20)  #100
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=5)  #4096
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', default='cuda:0')  #cuda:0, cpu
    parser.add_argument('--save_dir', default='./')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--ckpt_path', type=str, default=None)
    args = parser.parse_args()
    main(args.dataset_path, args.model_name, args.epoch, args.learning_rate, args.batch_size, args.weight_decay, args.device, args.save_dir, args.seed, args.ckpt_path)
#python run_kuairand_ctr.py --model_name DPRP --batch_size 10 --device cuda:0' --seed 1 --save_dir /root/Test/KDD/D-PRP/output 执行代码