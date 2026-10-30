"""Qt-independent decisions for reviewing a group_folder result."""


class GroupReview:
    def __init__(self, grouping, scores):
        self.groups = grouping['groups']
        self.scores = dict(scores)
        self.photos = {p['path']: p for g in self.groups for p in g['photos']}
        self.by_path = {p['path']: g for g in self.groups for p in g['photos']}
        self.defaults = {path for path, p in self.photos.items() if p['is_best']}
        self.kept = set(self.defaults)

    def is_kept(self, path):
        return path in self.kept

    def toggle(self, path):
        if path not in self.photos:
            raise KeyError(path)
        self.kept.symmetric_difference_update({path})

    def _groups(self, group_ids):
        return [g for g in self.groups if group_ids is None or g['group_id'] in group_ids]

    def discard_paths(self, group_ids=None):
        """group_ids 為 None 時是全部組，否則只算這些組。"""
        return [p['path'] for g in self._groups(group_ids) for p in g['photos']
                if p['path'] not in self.kept]

    def all_discarded_groups(self, group_ids=None):
        return [g['group_id'] for g in self._groups(group_ids)
                if not any(self.is_kept(p['path']) for p in g['photos'])]

    def groups_with_discards(self, exclude=()):
        """有照片要丟、而且不在 exclude 裡的組（用來找還沒看過的組）。"""
        return [g['group_id'] for g in self.groups if g['group_id'] not in exclude
                and any(not self.is_kept(p['path']) for p in g['photos'])]

    def adjusted_groups(self):
        return {self.by_path[p]['group_id'] for p in self.kept ^ self.defaults}

    def reason(self, path):
        group = self.by_path[path]
        best = group['best_path']
        a, t, overall = self.scores[path]
        ba, bt, bo = self.scores[best]
        if self.photos[path]['is_best']:
            if t == max(self.scores[p['path']][1] for p in group['photos']):
                return '推薦：技術分最高'
            if a == max(self.scores[p['path']][0] for p in group['photos']):
                return '推薦：構圖／美感最好'
            return '推薦：綜合分最高'
        return f'比推薦低 {max(0, bo - overall):.1f} 分' + ('、技術分較低' if bt - t >= 10 else '')

    def summary(self):
        return len(self.kept), len(self.photos) - len(self.kept)
