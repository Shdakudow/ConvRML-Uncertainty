import unittest
from collections import Counter
from scripts.prepare_large_intervals import extension_slots, COUNTS, OFFSETS


class LargeSplitTest(unittest.TestCase):
    def test_expansion_preserves_old_rows_and_fills_all_new_rows_once(self):
        original={'train':[None]*20000,'val':[None]*2500,'test':[None]*2500}
        slots=list(extension_slots(original))
        self.assertEqual(Counter(role for role,index in slots),
                         {'train':40000,'val':2500,'test':2500})
        self.assertEqual(slots[0],('val',62500))
        self.assertEqual(slots[2500],('test',67500))
        self.assertEqual(slots[5000],('train',20000))
        old={OFFSETS[role]+i for role,rows in original.items() for i in range(len(rows))}
        new={index for role,index in slots}
        self.assertEqual(len(new),45000)
        self.assertFalse(old & new)
        self.assertEqual(old | new,set(range(sum(COUNTS.values()))))


if __name__=='__main__':unittest.main()
