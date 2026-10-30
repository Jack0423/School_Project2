"""Inline review controller. The host owns its existing preview and results."""
from pathlib import Path
from PyQt6.QtCore import Qt, QObject, QTimer
from PyQt6.QtGui import QPixmap, QShortcut, QKeySequence
from PyQt6.QtWidgets import (QWidget, QLabel, QPushButton, QListWidget,
    QVBoxLayout, QHBoxLayout, QScrollArea, QCheckBox, QGraphicsOpacityEffect,
    QMessageBox, QSizePolicy)
from grouping_review import GroupReview
from grouping_thumbnails import ThumbnailCache


class GroupReviewView(QObject):
    def __init__(self, host, load_groups, get_scores, preview, delete, recompute,
                 body, left_panel, center, bottom, get_signature):
        super().__init__(host)
        self.host = host
        self.load_groups, self.get_scores = load_groups, get_scores
        self.preview, self.delete, self.recompute = preview, delete, recompute
        self.get_signature = get_signature
        self.source_signatures = {}
        self.left_panel = left_panel
        self.review = None
        self.active = False
        self.seen = set()
        self.cards = {}
        self.current_photo = 0
        self.cache = ThumbnailCache(self)
        self.cache.ready.connect(self.thumbnail_ready)
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self.show_preview)
        self.sidebar = QWidget()
        layout = QVBoxLayout(self.sidebar)
        self.count = QLabel()
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.burst = QCheckBox('只看連拍（同相機、5 秒內）')
        self.burst.toggled.connect(self.reload)
        self.groups = QListWidget()
        self.groups.currentRowChanged.connect(self.select_group)
        layout.addWidget(self.count)
        layout.addWidget(self.burst)
        layout.addWidget(self.status)
        layout.addWidget(self.groups, 1)
        body.insertWidget(0, self.sidebar, 17)
        self.strip = QWidget()
        strip_layout = QVBoxLayout(self.strip)
        self.heading = QLabel()
        strip_layout.addWidget(self.heading)
        hint = QLabel('← → 看照片　空白鍵 保留／丟掉　Enter／↓ 下一組　↑ 上一組　Esc 離開')
        hint.setWordWrap(True)
        strip_layout.addWidget(hint)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFixedHeight(250)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        strip_layout.addWidget(self.scroll)
        center.addWidget(self.strip)
        self.footer = QWidget()
        footer_layout = QHBoxLayout(self.footer)
        self.summary_label = QLabel()
        self.remove = QPushButton()
        self.remove.clicked.connect(self.delete_discarded)
        footer_layout.addWidget(self.summary_label, 1)
        footer_layout.addWidget(self.remove)
        bottom.addWidget(self.footer)
        self.shortcuts = []
        for key, action in [('Left', lambda: self.move_photo(-1)),
                            ('Right', lambda: self.move_photo(1)),
                            ('Space', self.toggle_current),
                            ('Return', lambda: self.move_group(1)),
                            ('Enter', lambda: self.move_group(1)),
                            ('Down', lambda: self.move_group(1)),
                            ('Up', lambda: self.move_group(-1)),
                            ('Escape', self.exit)]:
            shortcut = QShortcut(QKeySequence(key), host, activated=action)
            shortcut.setEnabled(False)
            self.shortcuts.append(shortcut)
        self._visible(False)

    def _visible(self, active):
        self.active = active
        if not active:
            self.sidebar.hide()
            self.strip.hide()
            self.footer.hide()
        for shortcut in self.shortcuts:
            shortcut.setEnabled(active)
        self.host.weight_slider.setEnabled(not active)
        if active:
            self.weight_tooltip = self.host.weight_slider.toolTip()
            self.host.weight_slider.setToolTip('離開整理模式後才能調整')
            self.preview_policy = self.host.preview.sizePolicy()
            self.host.preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
            self.host.preview.setMinimumSize(450, 220)
            self.progress_visibility = [(w, not w.isHidden()) for w in
                                        (self.host.progress_bar, self.host.cancel_btn)]
            for widget, _ in self.progress_visibility:
                widget.hide()
            self.saved_controls = [(w, w.isEnabled()) for w in
                                   (self.host.analyze_current_btn, self.host.analyze_all_btn,
                                    self.host.xmp_btn)]
            for control, _ in self.saved_controls:
                control.setEnabled(False)
        else:
            if hasattr(self, 'weight_tooltip'):
                self.host.weight_slider.setToolTip(self.weight_tooltip)
                self.host.preview.setSizePolicy(self.preview_policy)
                for widget, visible in self.progress_visibility:
                    widget.setVisible(visible)
                for control, enabled in self.saved_controls:
                    control.setEnabled(enabled)
            self.host.preview.setMinimumSize(650, 560)
        self.sidebar.setVisible(active)
        self.strip.setVisible(active)
        self.footer.setVisible(active)
        self.left_panel.setVisible(not active)
        self.host.group_btn.setText('離開整理模式' if active else '相似照片分組')
        if hasattr(self.host, 'group_review'):
            self.host._update_primary_action()

    def enter(self):
        # Reopening must never show thumbnails from an older file at this path.
        self.cache.clear()
        window_size = self.host.size()
        self._visible(True)
        self.burst.blockSignals(True)
        self.burst.setChecked(False)
        self.burst.blockSignals(False)
        self.reload()
        # Showing the strip can temporarily expand the old preview minimum.
        # Respect the user's window size once review constraints are in place.
        if not self.host.isMaximized():
            self.host.layout().invalidate()
            self.host.layout().activate()
            self.host.resize(window_size)

    def exit(self, refresh=True):
        self.preview_timer.stop()
        self._visible(False)
        self.review = None
        if refresh:
            self.host._refresh_workflow_state()

    def folder_changed(self):
        self.exit(refresh=False)
        self.cache.clear()

    def reload(self, *args, after_delete=False):
        self.preview_timer.stop()
        self.groups.blockSignals(True)
        self.groups.clear()
        self.seen.clear()
        try:
            result = self.load_groups('burst' if self.burst.isChecked() else 'similar')
            scores = {p['path']: self.get_scores(p['path'])
                      for g in result['groups'] for p in g['photos']}
            self.review = GroupReview(result, scores)
            self.source_signatures = {path: self.get_signature(path) for path in scores}
            # Reuse the grouping needed by the view to refresh cached counts.
            self.host._refresh_workflow_state(
                grouping=result if not self.burst.isChecked() else None,
                update_status=False)
            self.host._update_group_status(result)
            messages = []
            if result['not_ready']:
                messages.append(f"{len(result['not_ready'])} 張尚未準備好，請離開整理模式後批次分析。")
            if result['no_capture_time']:
                messages.append(f"{len(result['no_capture_time'])} 張缺少可用拍攝時間或相機資訊，未納入連拍分組；請重新批次分析。")
            if result['exiftool_missing']:
                messages.append('找不到 ExifTool，請安裝後重新啟動程式並執行批次分析。')
            if not result['groups']:
                messages.insert(0, '沒有可整理的照片組。')
            self.status.setText('\n'.join(messages))
            for group in self.review.groups:
                self.groups.addItem(f"第 {group['group_id']} 組（{group['count']} 張）")
        except Exception as exc:
            self.review = None
            self.status.setText(f'分組失敗：{exc}')
        finally:
            self.groups.blockSignals(False)
        self.clear_cards()
        self.update_summary()
        if self.review and self.review.groups:
            self.groups.setCurrentRow(0)
        elif after_delete and self.review is not None:
            self.exit(refresh=False)
            self.host._update_group_status()
            QMessageBox.information(self.host, '照片分組', '已沒有可整理的照片組，已離開整理模式。')

    def clear_cards(self):
        self.cards = {}
        container = QWidget()
        self.card_layout = QHBoxLayout(container)
        self.card_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        old = self.scroll.takeWidget()
        if old:
            old.deleteLater()
        self.scroll.setWidget(container)
        self.heading.setText('')

    def select_group(self, row):
        if self.review is None or not 0 <= row < len(self.review.groups):
            return
        group = self.review.groups[row]
        self.seen.add(group['group_id'])
        self.current_photo = 0
        self.clear_cards()
        self.heading.setText(f"第 {group['group_id']} 組 · {group['count']} 張")
        for photo in group['photos']:
            path = photo['path']
            card = QWidget()
            card.setObjectName("reviewPhotoCard")
            card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            card.setFixedWidth(220)
            layout = QVBoxLayout(card)
            layout.setContentsMargins(6, 6, 6, 6)
            layout.setSpacing(3)
            image = QPushButton('載入中…')
            image.setFixedSize(200, 110)
            image.setStyleSheet('background:#555; border-radius:8px;')
            image.clicked.connect(lambda checked=False, p=path: self.choose_photo(p))
            toggle = QPushButton()
            toggle.setFixedHeight(28)
            toggle.clicked.connect(lambda checked=False, p=path: self.toggle(p))
            score = QLabel(f'綜合分 {self.review.scores[path][2]:.1f}')
            score.setFixedHeight(18)
            name = QLabel(Path(path).name)
            name.setFixedHeight(18)
            name.setTextFormat(Qt.TextFormat.PlainText)
            name.setToolTip(path)
            reason = QLabel(self.review.reason(path))
            reason.setWordWrap(True)
            reason.setFixedHeight(36)
            for widget in (image, toggle, score, name, reason):
                layout.addWidget(widget)
            self.card_layout.addWidget(card)
            self.cards[path] = card, image, toggle
            self.cache.request(path)
        self.update_summary()
        self.choose_photo(group['photos'][0]['path'])

    def thumbnail_ready(self, path, image, error):
        if path not in self.cards:
            return
        button = self.cards[path][1]
        if image.isNull():
            button.setText('縮圖載入失敗')
            button.setToolTip(error)
        else:
            from PyQt6.QtGui import QIcon
            button.setText('')
            button.setIcon(QIcon(QPixmap.fromImage(image)))
            button.setIconSize(button.size())

    def choose_photo(self, path):
        group = self.review.groups[self.groups.currentRow()]
        self.current_photo = [p['path'] for p in group['photos']].index(path)
        self.selected_path = path
        self.update_cards()
        self.scroll.ensureWidgetVisible(self.cards[path][0])
        # Coalesce rapid arrow presses before the existing large preview loader runs.
        self.preview_timer.start(40)

    def show_preview(self):
        if self.active and self.review and self.selected_path in self.review.photos:
            self.preview(self.selected_path)

    def update_cards(self):
        if not self.review:
            return
        for path, (card, image, toggle) in self.cards.items():
            kept = self.review.is_kept(path)
            toggle.setText(('保留' if kept else '丟掉') +
                           (' · 推薦' if self.review.photos[path]['is_best'] else ''))
            toggle.setStyleSheet(('background:#15582f;color:#b4ffcf;' if kept
                                 else 'background:#393939;color:#ccc;') + 'padding:2px 8px;')
            effect = QGraphicsOpacityEffect(image)
            effect.setOpacity(1 if kept else .35)
            image.setGraphicsEffect(effect)
            border = '#b8c2cc' if path == getattr(self, 'selected_path', None) else 'transparent'
            card.setStyleSheet(
                f'QWidget#reviewPhotoCard {{ border:2px solid {border}; border-radius:6px; }}')

    def toggle(self, path):
        self.review.toggle(path)
        self.update_summary()

    def toggle_current(self):
        if self.review and self.review.groups:
            self.toggle(self.selected_path)

    def move_photo(self, delta):
        if self.review and self.review.groups:
            photos = self.review.groups[self.groups.currentRow()]['photos']
            index = max(0, min(len(photos) - 1, self.current_photo + delta))
            self.choose_photo(photos[index]['path'])

    def move_group(self, delta):
        if self.review and self.review.groups:
            self.groups.setCurrentRow(max(0, min(self.groups.count() - 1,
                                                self.groups.currentRow() + delta)))

    def update_summary(self):
        n = len(self.review.groups) if self.review else 0
        kept, discarded = self.review.summary() if self.review else (0, 0)
        self.count.setText(f'共 {n} 組\u30fb已看 {len(self.seen)} 組')
        self.summary_label.setText(f'全部 {n} 組：保留 {kept} 張、丟掉 {discarded} 張')
        self.remove.setText(f'移到資源回收筒（{discarded}）')
        self.remove.setEnabled(discarded > 0)
        if self.review:
            adjusted = self.review.adjusted_groups()
            for i, group in enumerate(self.review.groups):
                gid = group['group_id']
                self.groups.item(i).setText(('\u2713 ' if gid in self.seen else '') +
                    f"第 {gid} 組（{group['count']} 張）" + (' · 已調整' if gid in adjusted else ''))
        self.update_cards()

    def ask_unseen(self, unseen_groups, unseen_photos, seen_photos):
        """回傳 'seen'（只刪已看過的組）、'all'（依然全部刪除）或 None（返回）。"""
        box = QMessageBox(QMessageBox.Icon.Warning, '還有沒看過的組',
                          f'還有 {unseen_groups} 組沒看過，裡面有 {unseen_photos} 張'
                          '預設會被丟掉。\n要怎麼處理？', parent=self.host)
        only_seen = box.addButton(f'只刪除已看過的（{seen_photos} 張）',
                                  QMessageBox.ButtonRole.AcceptRole)
        only_seen.setEnabled(seen_photos > 0)
        everything = box.addButton(f'依然全部刪除（{seen_photos + unseen_photos} 張）',
                                   QMessageBox.ButtonRole.DestructiveRole)
        back = box.addButton('返回', QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(back)
        box.setEscapeButton(back)
        box.exec()
        clicked = box.clickedButton()
        if clicked is only_seen:
            return 'seen'
        if clicked is everything:
            return 'all'
        return None

    def delete_discarded(self):
        if not self.review:
            return
        paths = self.review.discard_paths()
        if not paths:
            return
        # 預設每組非推薦都是丟掉；還沒看過的組先問，避免一鍵連沒看的一起刪掉。
        groups = None
        unseen = self.review.groups_with_discards(exclude=self.seen)
        if unseen:
            seen_paths = self.review.discard_paths(self.seen)
            choice = self.ask_unseen(len(unseen), len(paths) - len(seen_paths),
                                     len(seen_paths))
            if choice is None:
                return
            if choice == 'seen':
                groups, paths = set(self.seen), seen_paths
        yes, no = QMessageBox.StandardButton.Yes, QMessageBox.StandardButton.No
        empty = self.review.all_discarded_groups(groups)
        if empty and QMessageBox.question(self.host, '整組全部丟掉',
                '第 ' + '、'.join(map(str, empty)) + ' 組全部都會丟掉，確定嗎？',
                yes | no, no) != yes:
            return
        names = '\n'.join(Path(p).name for p in paths[:8])
        if len(paths) > 8:
            names += f'\n……共 {len(paths)} 張'
        if QMessageBox.question(self.host, '刪除照片',
                f'要把這 {len(paths)} 張照片移到資源回收筒嗎？\n'
                'RAW 旁邊的 .xmp 也會一起移走。之後可以從資源回收筒還原。\n\n' + names,
                yes | no, no) != yes:
            return
        self.preview_timer.stop()
        # Confirmations can stay open while an editor replaces a file. Check all
        # members, including the photo kept as the reference, before deleting.
        changed = []
        for path, signature in self.source_signatures.items():
            try:
                current_scores = self.get_scores(path)
            except (TypeError, KeyError):
                current_scores = None
            except Exception as exc:
                QMessageBox.warning(self.host, '無法確認照片資料',
                    f'未執行刪除，無法確認最新分析結果：{exc}')
                return
            if (signature is None or self.get_signature(path) != signature
                    or current_scores != self.review.scores[path]):
                changed.append(path)
        if changed:
            self.cache.clear()
            self.reload()
            QMessageBox.warning(self.host, '照片資料已變更',
                '整理期間有照片變更或分析結果更新，已取消本次刪除並重新整理。\n'
                '請確認最新照片；需要重新分析時，請先離開整理模式。')
            return
        try:
            moved, failed = self.delete(paths)
        except Exception as exc:
            QMessageBox.warning(self.host, '刪除照片失敗', str(exc))
            return
        self.recompute()
        self.host.refresh_list()
        if (self.host.current_file_path in paths
                and not Path(self.host.current_file_path).is_file()):
            self.host.current_file_path = None
            self.host.current_image_buffer = None
            self.host.preview.clearPhoto()
            self.host.file_label.setText('目前尚未選取照片')
            self.host.result_label.setText('尚未選取照片')
        # Deleted paths can reappear with new file contents; don't reuse stale images.
        self.cache.clear()
        message = f'已移到資源回收筒：{moved} 張。'
        if failed:
            message += f'\n未能移走 {len(failed)} 個檔案（含 XMP）：\n' + '\n'.join(
                f'{name}：{reason}' for name, reason in failed[:8])
        QMessageBox.information(self.host, '刪除照片', message)
        self.reload(after_delete=True)
