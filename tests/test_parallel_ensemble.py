"""Verify checkpoint migration preserves each member's AdamW updates."""
import copy
import tempfile
from pathlib import Path
import unittest
import json
from unittest.mock import patch
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ensemble_jobs import member_batch_order, split_member_checkpoint, split_checkpoint, ready_epochs
from models.ensemble import DeepEnsemble, DeepEnsembleMember
from train import train_epoch, rng_state


class ParallelEnsembleTests(unittest.TestCase):
    def test_batch_membership_shared_order_independent_and_repeatable(self):
        loader = DataLoader(TensorDataset(torch.arange(103)), batch_size=10, shuffle=True)
        orders = [member_batch_order(loader, 4, i) for i in range(10)]
        canonical = lambda order: sorted(tuple(batch) for batch in order)

        self.assertTrue(all(canonical(order) == canonical(orders[0]) for order in orders))
        self.assertEqual(len({str(order) for order in orders}), 10)
        self.assertEqual(orders[3], member_batch_order(loader, 4, 3))
        self.assertNotEqual(canonical(orders[0]), canonical(member_batch_order(loader, 5, 0)))

    def test_split_members_preserves_epoch_weights_and_optimizer(self):
        torch.manual_seed(777)
        make = lambda: DeepEnsemble([DeepEnsembleMember(nn.Conv2d(3, 6, 1)) for _ in range(10)])
        model = make()
        loader = DataLoader(
            TensorDataset(torch.rand(9, 3, 4, 4), torch.rand(9, 3, 4, 4)),
            batch_size=4, shuffle=True,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
        train_epoch(model, loader, optimizer, 'cpu', 0, 150)
        saved = copy.deepcopy(dict(
            model=model.state_dict(), optimizer=optimizer.state_dict(),
            epoch=1, global_step=3, history=[],
            best_checkpoint=None, height=4, width=4, dataset_size=9, rng=rng_state(),
        ))

        for index in range(10):
            split = split_member_checkpoint(saved, index)
            single = DeepEnsemble([DeepEnsembleMember(nn.Conv2d(3, 6, 1))])
            single.load_state_dict(split['model'])
            opt = torch.optim.AdamW(single.parameters())
            opt.load_state_dict(split['optimizer'])

            torch.testing.assert_close(
                single.members[0].state_dict(), model.members[index].state_dict(), rtol=0, atol=0,
            )
            reference = optimizer.state_dict()
            width = len(list(single.parameters()))
            for own in range(width):
                torch.testing.assert_close(
                    opt.state_dict()['state'][own],
                    reference['state'][index * width + own],
                    rtol=0, atol=0,
                )
            self.assertEqual(split['epoch'], 1)
            self.assertEqual(split['global_step'], 3)

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'old.pt'
            output = Path(folder) / 'parallel'
            torch.save(saved, source)
            split_checkpoint(source, output)
            self.assertTrue(all((output / f'member_{i:02d}' / 'latest.pt').exists() for i in range(10)))
            self.assertEqual(ready_epochs(output), [])

    def test_member_entrypoint_saves_epoch_snapshots_without_single_member_validation(self):
        import train
        dataset = TensorDataset(torch.rand(4, 3, 4, 4), torch.rand(4, 3, 4, 4))
        loader = DataLoader(dataset, batch_size=2, shuffle=True)

        with tempfile.TemporaryDirectory() as folder:
            args = ['train.py', '--manifest', 'unused.json', '--model', 'ensemble',
                    '--ensemble-member', '3', '--output', folder]
            with (
                patch('sys.argv', args),
                patch.object(train, 'get_loaders', return_value=(loader, loader)),
                patch.object(train, 'ConvRecon', side_effect=lambda *a, **kw: nn.Conv2d(3, 6, 1)),
                patch.object(train, 'EPOCHS', 2),
                patch.object(train, 'validate', side_effect=AssertionError('Validation must combine all ten members.')),
                patch('builtins.print'),
            ):
                train.main()

            for epoch in (1, 2):
                saved = torch.load(Path(folder) / f'epoch_{epoch:03d}.pt', weights_only=False)
                self.assertEqual(saved['epoch'], epoch)
                self.assertEqual(saved['ensemble_member'], 3)

            saved = torch.load(Path(folder) / 'latest.pt', weights_only=False)
            self.assertEqual(saved['global_step'], 4)
            self.assertEqual(saved['epoch'], 2)

    def test_evaluation_combines_members_and_keeps_cumulative_best(self):
        from ensemble_jobs import evaluate_ready
        make = lambda *args: DeepEnsemble([DeepEnsembleMember(nn.Conv2d(3, 6, 1)) for _ in range(10)])
        template = make()

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'history.json').write_text(json.dumps([dict(epoch=i, ssim=.5) for i in range(1, 4)]))
            torch.save(dict(model=template.state_dict(), epoch=3, validation_ssim=.5), root / 'best.pt')

            for epoch in (4, 5):
                for index in range(10):
                    member = DeepEnsemble([copy.deepcopy(template.members[index])])
                    path = root / f'member_{index:02d}'
                    path.mkdir(exist_ok=True)
                    torch.save(
                        dict(model=member.state_dict(), epoch=epoch, height=4, width=4, dataset_size=9),
                        path / f'epoch_{epoch:03d}.pt',
                    )

            self.assertEqual(ready_epochs(root), [4, 5])
            scores = iter((.7, .65))
            image = torch.rand(2, 3, 4, 4)

            def validate(model, *args):
                torch.testing.assert_close(model.center(image), template.center(image), rtol=0, atol=0)
                return dict(ssim=next(scores), psnr=20., lpips=.2)

            with (
                patch('dataloader.get_loaders', return_value=(None, None)),
                patch('models.make_model', side_effect=make),
                patch.object(nn.Module, 'cuda', lambda self: self),
                patch('train.validate', side_effect=validate),
                patch('torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity', return_value=nn.Identity()),
            ):
                evaluate_ready(Path('unused.json'), root)

            selected = torch.load(root / 'best_through_epoch_005.pt', weights_only=False)
            self.assertEqual(selected['epoch'], 4)
            self.assertEqual(selected['validation_ssim'], .7)
            self.assertEqual(
                [x['epoch'] for x in json.loads((root / 'history.json').read_text())],
                [1, 2, 3, 4, 5],
            )
            self.assertEqual(ready_epochs(root), [])


if __name__ == '__main__':
    unittest.main()
