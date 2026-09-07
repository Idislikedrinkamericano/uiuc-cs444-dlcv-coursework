import os
import torch
import sys
import tqdm
import time
import logging
from absl import app, flags
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import yaml
from model import PixMoModel, Trainer, inference
from datasets import get_pixmo_data
from torch.optim.lr_scheduler import StepLR

FLAGS = flags.FLAGS
flags.DEFINE_string('exp_name', 'pixmo_v2',
                    'Experiment name; must match key in config.yaml')
flags.DEFINE_string('output_dir', 'runs_pixmo', 'Output directory for logs and checkpoints')
flags.DEFINE_string('data_dir', './pixmo_data', 'Directory with PixMo data')

# ------------------------- Logging setup -------------------------
def setup_logging():
    log_formatter = logging.Formatter(
        '%(asctime)s: %(levelname)s %(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logging.getLogger().handlers = []
    if len(logging.getLogger().handlers) == 0:
        logging.getLogger().addHandler(console_handler)
    logging.getLogger().setLevel(logging.INFO)

def logger(tag, value, global_step):
    if tag == '':
        logging.info('')
    else:
        logging.info(f'  {tag:>15s} [{global_step:07d}]: {value:5f}')

class SummaryWriterWithPrinting(SummaryWriter):
    def add_scalar(self, tag, value, global_step):
        super(SummaryWriterWithPrinting, self).add_scalar(tag, value, global_step)
        logger(tag, value, global_step)

# ------------------------- Config Loader -------------------------
def get_config(exp_name):
    dir_name = f'{FLAGS.output_dir}'
    encoder_registry = {
        'PixMoModel': PixMoModel,
    }
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)[exp_name]

    lr = config['lr']
    epochs = config['epochs']
    optimizer = config['optimizer']
    net_class = encoder_registry[config['net_class']]
    batch_size = config['batch_size']
    num_classes = config['num_classes']

    return net_class, (num_classes,), dir_name, (optimizer, lr, epochs, batch_size)

# ------------------------- Main -------------------------
def main(_):
    setup_logging()
    torch.set_num_threads(4)
    torch.manual_seed(42)

    print(f"\n🚀 Running experiment: {FLAGS.exp_name}")
    print(f"📁 Output directory: {FLAGS.output_dir}\n")

    net_class, (num_classes,), dir_name, \
        (optimizer, lr, epochs, batch_size) = get_config(FLAGS.exp_name)

    train_data = get_pixmo_data(FLAGS.data_dir, 'train')
    val_data = get_pixmo_data(FLAGS.data_dir, 'val')
    test_data = get_pixmo_data(FLAGS.data_dir, 'test')

    train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=4, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, num_workers=4, shuffle=False)

    os.makedirs(dir_name, exist_ok=True)
    tmp_file_name = os.path.join(dir_name, 'best_model.pth')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Using device: {device}")

    writer = SummaryWriterWithPrinting(f'{dir_name}', flush_secs=10)

    model = net_class(num_classes)
    model.to(device)

    trainer = Trainer(model, train_dataloader, val_dataloader, writer,
                      optimizer=optimizer, lr=lr, epochs=epochs, device=device)

    # Add learning rate scheduler
    trainer.scheduler = StepLR(trainer.optimizer, step_size=max(1, epochs // 3), gamma=0.5)

    best_val_acc, best_epoch = trainer.train(model_file_name=tmp_file_name)
    print(f"\n✅ Training complete! lr={lr:0.7f}, best_val_acc={best_val_acc:.4f}, best_epoch={best_epoch}\n")

    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    model.load_state_dict(torch.load(tmp_file_name, map_location=device))
    inference(test_dataloader, model, device, result_path=os.path.join(dir_name, 'test_pixmo.txt'))

    print("📊 Test predictions saved to:", os.path.join(dir_name, 'test_pixmo.txt'))

if __name__ == '__main__':
    app.run(main)
