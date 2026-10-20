import unittest
from grouping_review import GroupReview


def fixture():
    return {'groups': [dict(group_id=1, count=3, best_path='a', photos=[
        dict(path=p, is_best=p == 'a') for p in 'abc']),
        dict(group_id=2, count=2, best_path='d', photos=[
        dict(path=p, is_best=p == 'd') for p in 'de'])]}


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.scores = dict(a=(70, 80, 76), b=(80, 70, 74), c=(60, 71, 65),
                           d=(80, 60, 72), e=(70, 70, 70))
        self.review = GroupReview(fixture(), self.scores)

    def test_defaults(self):
        self.assertEqual([p for p in 'abcde' if self.review.is_kept(p)], ['a', 'd'])
        self.assertEqual(self.review.summary(), (2, 3))

    def test_toggle_multiple_kept_and_revert(self):
        self.review.toggle('b')
        self.assertEqual(self.review.discard_paths(), ['c', 'e'])
        self.assertEqual(self.review.summary(), (3, 2))
        self.assertEqual(self.review.adjusted_groups(), {1})
        self.review.toggle('b')
        self.assertEqual(self.review.adjusted_groups(), set())
        self.review.toggle('a')
        self.review.toggle('b')
        self.assertEqual(self.review.discard_paths(), ['a', 'c', 'e'])

    def test_all_discarded(self):
        self.review.toggle('a')
        self.assertEqual(self.review.all_discarded_groups(), [1])
        self.review.toggle('d')
        self.assertEqual(self.review.all_discarded_groups(), [1, 2])

    def test_reasons_and_blur_threshold(self):
        self.assertEqual(self.review.reason('a'), '推薦：技術分最高')
        self.assertEqual(self.review.reason('d'), '推薦：構圖／美感最好')
        self.assertEqual(self.review.reason('b'), '比推薦低 2.0 分、技術分較低')
        self.assertEqual(self.review.reason('c'), '比推薦低 11.0 分')
        scores = dict(self.scores, a=(70, 75, 76))
        self.assertEqual(GroupReview(fixture(), scores).reason('a'), '推薦：技術分最高')
        scores['c'] = (90, 90, 65)
        self.assertEqual(GroupReview(fixture(), scores).reason('a'), '推薦：綜合分最高')

    def test_invalid_path_does_not_affect_summary(self):
        with self.assertRaises(KeyError):
            self.review.toggle('unknown')
        self.assertEqual(self.review.summary(), (2, 3))
