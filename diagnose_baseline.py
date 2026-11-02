"""Why does the end-to-end GRU baseline (NeuralBaseline) score IOU-M ~ 0? Three quick diagnostics (about 2 minutes).

1. Its final training loss equals the loss of "move nothing" -> it sits on the trivial plateau.
2. Oracle subject: telling it the true subject (diagnostic only) lets it learn to move boxes.
3. Auxiliary subject supervision (diagnostic only) lets the same architecture ground subjects.
So the architecture and code can represent the task; what fails is learning the subject
grounding from the final-box loss alone.

    python diagnose_baseline.py
"""
import torch
import torch.nn.functional as F

from nsrmp.baseline import NeuralBaseline
from nsrmp.batch import collate
from nsrmp.data import make_dataset
from nsrmp.metrics import evaluate
from nsrmp.model import apply_move
from nsrmp.train import loss_fn, train

EPOCHS = 30


class OracleSubjectBaseline(NeuralBaseline):
    """Single-step only: the subject weights are replaced by the gold one-hot."""

    def forward(self, batch):
        sentence = self.encode(batch.tokens[:, 0], batch.lengths[:, 0])
        subj = F.one_hot(batch.subjects[:, 0], batch.boxes.shape[1]).float()
        ref = self._attend(self.reference_head, batch.feats, batch.boxes, sentence, batch.mask)
        action = F.softmax(self.action_head(sentence), -1) @ self.sim.action_emb.weight
        return apply_move(self.sim, batch.boxes, subj, ref, action), subj[:, None]


def main():
    train_set = collate(make_dataset(4000, seed=0))
    test_set = collate(make_dataset(1000, seed=1000))
    do_nothing = loss_fn(test_set.boxes, test_set.final, test_set.mask).item()
    print(f"loss of 'move nothing' on the test set: {do_nothing:.2f}")

    torch.manual_seed(0)
    baseline = NeuralBaseline()
    history = train(baseline, train_set, EPOCHS, seed=0)
    print(f"1. baseline final training loss {history[-1]:.2f}; test {evaluate(baseline, test_set)}")

    torch.manual_seed(0)
    oracle = OracleSubjectBaseline()
    train(oracle, train_set, EPOCHS, seed=0)
    print(f"2. oracle subject: {evaluate(oracle, test_set)}")

    torch.manual_seed(0)
    aux = NeuralBaseline()
    optimizer = torch.optim.Adam(aux.parameters(), lr=3e-3)
    for _ in range(EPOCHS):
        order = torch.randperm(len(train_set))
        for start in range(0, len(order), 64):
            batch = train_set.select(order[start:start + 64])
            pred, subj_w = aux(batch)
            gold_w = subj_w[:, 0].gather(1, batch.subjects[:, :1])[:, 0]
            loss = loss_fn(pred, batch.final, batch.mask) - torch.log(gold_w + 1e-6).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print(f"3. extra subject-supervision loss: {evaluate(aux, test_set)}")


if __name__ == "__main__":
    main()
