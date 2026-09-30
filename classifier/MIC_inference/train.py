import argparse
import torch
from torch import nn
import pandas as pd
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from model import REG
import matplotlib.pyplot as plt
from seq_dataloader import _get_train_data_loader, _get_test_data_loader, freeze
from config import DATA_DIR, RESULT_DIR

def train(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    train_loader = _get_train_data_loader(args.batch_size, args.train_dir, args.train_frac)
    test_loader = _get_test_data_loader(500, args.test_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = REG()
    freeze(model, args.frozen_layers)
    model = model.to(device)

    optimizer = optim.AdamW(
        filter(lambda x: x.requires_grad is not False,
        model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay)

    loss_fn = nn.MSELoss()
    
    # Initialize storage lists
    mse_list = []
    r2_list = []
    pcc_list = []
    test_mse_list = []
    test_r2_list = []
    test_pcc_list = []
    
    # Initialize best model parameters
    best_model_state = None
    best_test_mse = float('inf')
    best_test_pcc = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        model.train()
        train_predict_list = []
        train_target_list = []

        for batch in train_loader:
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['targets'].to(device)

            predict_MIC, self_prot_bert = model(b_input_ids, attention_mask=b_input_mask)
            loss = loss_fn(predict_MIC.view(-1), b_labels.float().view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            train_predict_list.append(predict_MIC.cpu().data.numpy())
            train_target_list.append(b_labels.cpu())
        
        # Training set evaluation
        train_predict_list = np.concatenate(train_predict_list).flatten()
        train_target_list = np.concatenate(train_target_list).flatten()
        mse_epoch = mean_squared_error(train_predict_list, train_target_list)
        r2_epoch = r2_score(train_predict_list, train_target_list)
        pcc_epoch = pearsonr(train_predict_list, train_target_list)[0]
        
        mse_list.append(mse_epoch)
        r2_list.append(r2_epoch)
        pcc_list.append(pcc_epoch)

        # Test set evaluation
        model.eval()
        test_predict_list = []
        test_target_list = []
        with torch.no_grad():
            for batch in test_loader:
                b_input_ids = batch['input_ids'].to(device)
                b_input_mask = batch['attention_mask'].to(device)
                b_labels = batch['targets'].to(device)
                predict_MIC, _ = model(b_input_ids, attention_mask=b_input_mask)
                test_predict_list.append(predict_MIC.cpu().data.numpy())
                test_target_list.append(b_labels.cpu())
        
        test_predict_list = np.concatenate(test_predict_list).flatten()
        test_target_list = np.concatenate(test_target_list).flatten()
        test_mse = mean_squared_error(test_predict_list, test_target_list)
        test_r2 = r2_score(test_predict_list, test_target_list)
        test_pcc = pearsonr(test_predict_list, test_target_list)[0]
        
        test_mse_list.append(test_mse)
        test_r2_list.append(test_r2)
        test_pcc_list.append(test_pcc)
        
        # Save best model
        if test_mse < best_test_mse:
            best_test_mse = test_mse
            best_test_pcc = test_pcc
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, args.model_save_path)

        print(f"epoch: {epoch}, train_mse = {mse_epoch:.4f}, train_r2 = {r2_epoch:.4f}, "
              f"train_pcc: {pcc_epoch:.4f}, test_mse = {test_mse:.4f}, "
              f"test_r2 = {test_r2:.4f}, test_pcc: {test_pcc:.4f}")

    # Save training results
    result_dict = {
        "seed": args.seed,
        "epoch": list(range(1, args.epochs+1)),
        "train_mse": mse_list,
        "train_r2": r2_list,
        "train_pcc": pcc_list,
        "test_mse": test_mse_list,
        "test_r2": test_r2_list,
        "test_pcc": test_pcc_list,
        "best_mse": best_test_mse,
        "best_pcc": best_test_pcc,
        "best_epoch": best_epoch
    }
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(args.result_dir, index=False)
    
    # Plot and save loss trend charts
    plt.figure(figsize=(15, 10))
    
    # MSE trend chart
    plt.subplot(2, 2, 1)
    plt.plot(range(1, args.epochs+1), mse_list, 'b-', label='Train MSE')
    plt.plot(range(1, args.epochs+1), test_mse_list, 'r-', label='Test MSE')
    plt.scatter([best_epoch+1], [best_test_mse], color='red', s=100, zorder=5, 
                label=f'Best Test MSE: {best_test_mse:.4f}')
    plt.title('MSE Trend')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    
    # R2 trend chart
    plt.subplot(2, 2, 2)
    plt.plot(range(1, args.epochs+1), r2_list, 'b-', label='Train R2')
    plt.plot(range(1, args.epochs+1), test_r2_list, 'r-', label='Test R2')
    plt.title('R2 Score Trend')
    plt.xlabel('Epoch')
    plt.ylabel('R2 Score')
    plt.legend()
    plt.grid(True)
    
    # PCC trend chart
    plt.subplot(2, 2, 3)
    plt.plot(range(1, args.epochs+1), pcc_list, 'b-', label='Train PCC')
    plt.plot(range(1, args.epochs+1), test_pcc_list, 'r-', label='Test PCC')
    plt.scatter([best_epoch+1], [best_test_pcc], color='red', s=100, zorder=5, 
                label=f'Best Test PCC: {best_test_pcc:.4f}')
    plt.title('PCC Trend')
    plt.xlabel('Epoch')
    plt.ylabel('Pearson Correlation')
    plt.legend()
    plt.grid(True)
    
    # Save chart
    plt.tight_layout()
    plt.savefig(args.plot_save_path)
    plt.close()
    
    return args.seed, best_epoch, best_test_mse, best_test_pcc, test_r2_list[best_epoch]

if __name__ == "__main__":
    max_num_split = 1
    max_num_seed = 1
    for i in range(max_num_split):
        train_path = str(DATA_DIR / "train-EC.csv")
        test_path = str(DATA_DIR / "test-EC.csv")
        best_path = str(RESULT_DIR / "mse_loss_best_result-EC.csv")

        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        best_seed_list, best_mse_epoch_list, best_mse_list, best_mse_pcc_list, best_mse_r2_list = [], [], [], [], []
        
        for seed in range(max_num_seed):
            my_seed = seed
            result_path = str(RESULT_DIR / f"mse_loss_result-EC-{my_seed}.csv")
            model_save_path = str(RESULT_DIR / f"best_model-EC-{my_seed}.pth")
            plot_save_path = str(RESULT_DIR / f"training_plot-EC-{my_seed}.png")

            parser = argparse.ArgumentParser()
            parser.add_argument("--seed", type=int, default=my_seed, metavar="S", help="random seed (default: 43)")
            parser.add_argument("--train_dir", type=str, default=train_path)
            parser.add_argument("--train_frac", type=float, default=1.0)
            parser.add_argument("--test_dir", type=str, default=test_path)
            parser.add_argument("--result_dir", type=str, default=result_path)
            parser.add_argument("--model_save_path", type=str, default=model_save_path)
            parser.add_argument("--plot_save_path", type=str, default=plot_save_path)
            parser.add_argument(
                "--batch-size", type=int, default=12, metavar="N", help="input batch size for training (default: 12)"
            )
            parser.add_argument("--frozen_layers", type=int, default=0, metavar="NL",
                                help="number of frozen layers(default: 0)")
            parser.add_argument("--lr", type=float, default=1e-5, metavar="LR", help="learning rate (default: 1e-5)")
            parser.add_argument("--weight_decay", type=float, default=3e-3, metavar="M",
                                help="weight_decay (default: 0.003)")
            # Set number of training epochs to 15
            parser.add_argument("--epochs", type=int, default=15, metavar="N",
                                help="number of epochs to train (default: 15)")

            best_seed, best_epoch, best_test_mse, best_test_pcc, best_test_r2 = train(parser.parse_args())
            best_seed_list.append(best_seed)
            best_mse_epoch_list.append(best_epoch)
            best_mse_list.append(best_test_mse)
            best_mse_pcc_list.append(best_test_pcc)
            best_mse_r2_list.append(best_test_r2)

        best_result_dict = {
            "best_seed": best_seed_list,
            "best_epoch": best_mse_epoch_list,
            "best_mse": best_mse_list,
            "best_pcc": best_mse_pcc_list,
            "best_r2": best_mse_r2_list
        }
        best_result_df = pd.DataFrame(best_result_dict)
        best_result_df.to_csv(best_path, index=False)
    
    print("Training completed with visualizations and best model saved!")
