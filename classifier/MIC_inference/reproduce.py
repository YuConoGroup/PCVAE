from model import REG
import torch
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, kendalltau

from config import FINETUNE_MODEL_PKL, TEST_LOADER_PKL

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = str(FINETUNE_MODEL_PKL)
    model = REG()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(device)
    test_loader = torch.load(str(TEST_LOADER_PKL), weights_only=False)

    test_predict_list = []
    test_target_list = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            b_input_ids = batch['input_ids']
            b_input_mask = batch['attention_mask']
            b_labels = batch['targets']
            predict_MIC,_ = model(b_input_ids, attention_mask=b_input_mask)

            test_predict_list.extend(predict_MIC.data.numpy())
            test_target_list.extend(b_labels.data.numpy())

    test_predict_list = [item for sublist in test_predict_list for item in sublist]
    test_mse = mean_squared_error(test_predict_list, test_target_list)
    test_r2 = r2_score(test_predict_list, test_target_list)
    test_pcc = pearsonr(test_predict_list, test_target_list)[0]
    test_ktc = kendalltau(test_predict_list, test_target_list)[0]

    print('test_mse', '{:.4f}'.format(test_mse), 'test_r2', '{:.4f}'.format(test_r2),
              "test_pcc: ", '{:.4f}'.format(test_pcc), "test_ktc: ", '{:.4f}'.format(test_ktc))
