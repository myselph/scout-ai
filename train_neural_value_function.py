import argparse
import pickle
from common import StateAndScoreRecord
import torch
import torch.nn as nn
from neural_value_function import featurize, ScoutValueNet

parser = argparse.ArgumentParser()
parser.add_argument(
    "--files",
    type=lambda s: [str(item) for item in s.split(",")],
    help="pickle files",
    default=""
)
parser.add_argument(
    "--output",
    type=str,
    help="neural net weights output file (pth)",
    default="value_function.pth"
)
args = parser.parse_args()


def batch_generator(in_tensor: torch.Tensor, target_tensor: torch.Tensor,
                    batch_size: int):
    for i in range(0, len(in_tensor), batch_size):
        # Slice the list to get B tensors
        yield (in_tensor[i: i + batch_size],
               target_tensor[i: i + batch_size])



def main():
    # Load data from file(s)
    records: list[StateAndScoreRecord] = []
    for filename in args.files:
        with open(filename, "rb") as f:
            r = pickle.load(f)
            records = records + r
            print("loaded file " + filename)

    print(f"loaded {len(records)} records")

    # Featurize + normalize inputs and outputs
    inputs, outputs = featurize(records)
    del records
    split_index = int(len(inputs) / 10)
    train_in = inputs[split_index:]
    train_out = outputs[split_index:]
    val_in = inputs[:split_index]
    val_out = outputs[:split_index]
    del inputs, outputs
    print(f"Featurized inputs")

    model = ScoutValueNet()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0001,
        weight_decay=0.1)
    B = 1000
    for epoch in range(30):
        for (inputs, targets) in batch_generator(train_in, train_out, B):
            outputs = model(inputs).squeeze()
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_loss = loss_fn(model(train_in).squeeze(), train_out)
        val_loss = loss_fn(model(val_in).squeeze(), val_out)
        print(
            f"epoch {epoch}: train loss {
                train_loss:.3f}, val loss {
                val_loss:.3f}")
    if args.output:
        torch.save(model.state_dict(), args.output)


if __name__ == '__main__':
    main()
