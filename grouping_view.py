"""資料庫分組結果的前台視窗；不載入模型、不執行刪除。"""
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton,
    QLabel, QTreeWidget, QTreeWidgetItem,
)


class GroupingDialog(QDialog):
    def __init__(self, load_groups, non_best_paths, preview, select_paths, parent=None):
        super().__init__(parent)
        self.load_groups = load_groups
        self.non_best_paths = non_best_paths
        self.preview = preview
        self.select_paths = select_paths
        self.result = None
        self.setWindowTitle('相似照片與連拍分組')
        self.resize(850, 600)
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.mode = QComboBox()
        self.mode.addItem('相似照片', 'similar')
        self.mode.addItem('連拍候選', 'burst')
        controls.addWidget(self.mode)
        controls.addStretch()
        layout.addLayout(controls)
        self.status = QLabel()
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(['組別與照片', '綜合分數', '組內建議', '與推薦照片相似度'])
        self.tree.setColumnWidth(0, 330)
        layout.addWidget(self.tree)
        self.detail = QLabel('點選照片會更新主視窗的預覽與分析結果。組內建議保留與主清單的資料夾最佳不同。')
        self.detail.setTextFormat(Qt.TextFormat.PlainText)
        self.detail.setWordWrap(True)
        layout.addWidget(self.detail)
        self.select_button = QPushButton('選取每組非最佳並返回清單')
        self.close_button = QPushButton('返回清單')
        controls = QHBoxLayout()
        controls.addWidget(self.select_button)
        controls.addWidget(self.close_button)
        layout.addLayout(controls)
        self.mode.currentIndexChanged.connect(self.refresh)
        self.tree.itemClicked.connect(self.show_photo)
        self.select_button.clicked.connect(self.choose_non_best)
        self.close_button.clicked.connect(self.reject)
        self.refresh()

    def refresh(self):
        self.result = None
        self.tree.clear()
        self.select_button.setEnabled(False)
        try:
            result = self.load_groups(self.mode.currentData())
        except Exception as exc:
            self.status.setText(f'分組失敗：{exc}')
            return
        self.result = result
        missing = set(result['no_capture_time'])
        singles = [p for p in result['singles'] if p not in missing]
        messages = [f"找到 {len(result['groups'])} 組；未配對 {len(singles)} 張。"]
        if result['not_ready']:
            messages.append(f"{len(result['not_ready'])} 張尚未準備好，請返回按「開始批次分析」補齊資料。")
        if missing:
            messages.append(f'{len(missing)} 張缺少可用拍攝時間或相機資訊，未納入連拍分組；若尚未讀取，請重新批次分析。')
        if result['exiftool_missing']:
            messages.append('找不到 ExifTool，請安裝後重新啟動程式並執行批次分析。')
        self.status.setText('\n'.join(messages))
        for group in result['groups']:
            root = QTreeWidgetItem(self.tree, [f"第 {group['group_id']} 組（{group['count']} 張）"])
            for photo in group['photos']:
                item = QTreeWidgetItem(root, [
                    photo.get('file_name') or Path(photo['path']).name,
                    f"{photo['overall_score']:.2f}",
                    '本組建議保留' if photo['is_best'] else '待比較',
                    f"{photo['similarity_to_best']:.4f}",
                ])
                item.setData(0, Qt.ItemDataRole.UserRole, photo['path'])
                item.setToolTip(0, photo['path'])
            root.setExpanded(True)
        for title, paths in [('未配對照片', singles), ('尚未完成分析', result['not_ready']),
                             ('缺少拍攝時間或相機資訊', sorted(missing))]:
            if paths:
                root = QTreeWidgetItem(self.tree, [f'{title}（{len(paths)} 張）'])
                for path in paths:
                    item = QTreeWidgetItem(root, [Path(path).name])
                    item.setData(0, Qt.ItemDataRole.UserRole, path)
                    item.setToolTip(0, path)
        self.select_button.setEnabled(bool(self.non_best_paths(result)))

    def show_photo(self, item, column=0):
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if path:
            self.preview(path)
            self.detail.setText(f'主視窗已顯示：{Path(path).name}。可移動此視窗查看預覽。')

    def choose_non_best(self):
        # 重新讀取，以目前仍存在的照片與分數決定選取內容。
        self.refresh()
        if self.result is None:
            return
        paths = self.non_best_paths(self.result)
        if paths:
            self.select_paths(paths)
            self.accept()
