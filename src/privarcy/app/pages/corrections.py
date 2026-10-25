"""Correction Log (D4) + evidence-based rule refinement with validation, audit and rollback."""

from __future__ import annotations

import json
import time

from PyQt5.QtWidgets import QAbstractItemView, QHBoxLayout, QHeaderView, QMessageBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from ..widgets import Banner, Card, EmptyState, Page, Pill, button, clear_layout, label


class CorrectionsPage(Page):
    def __init__(self, app) -> None:
        super().__init__("Correction Log", "Every review decision, and the controlled process that can turn well-supported corrections into rule changes.")
        self.app = app
        self.banner = Banner()
        self.body.addWidget(self.banner)

        explain = Card(flat=True, padding=14)
        explain.add(label("Three different things:  1) Logging records each decision.  2) Rule refinement can lower one class's auto-redact "
                          "threshold — only with enough evidence, a passing validation, and your approval, and it can be rolled back.  "
                          "3) Model retraining is not done here: corrections never change the YOLO weights.", "Muted", wrap=True))
        self.body.addWidget(explain)

        self.rules_card = Card("Rule refinement")
        self.rules_body = QVBoxLayout()
        self.rules_body.setSpacing(10)
        self.rules_card.add(self.rules_body)
        self.body.addWidget(self.rules_card)

        self.log_card = Card("Decision history")
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["When", "Class", "Decision", "Confidence", "Frame", "Source"])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setMinimumHeight(300)
        self.log_card.add(self.table)
        self.empty = EmptyState("log", "No decisions logged yet", "Confirm or reject items in the Review Queue and they appear here.")
        self.log_card.add(self.empty)
        self.body.addWidget(self.log_card)
        self.body.addStretch()

    def on_show(self) -> None:
        svc = self.app.service
        records = svc.list_corrections(500)
        self.table.setRowCount(0)
        self.empty.setVisible(not records)
        self.table.setVisible(bool(records))
        for r in records:
            ctx = {}
            try:
                ctx = json.loads(r.context) if r.context else {}
            except ValueError:
                pass
            row = self.table.rowCount()
            self.table.insertRow(row)
            when = r.created_at[:16].replace("T", " ")
            vals = [when, r.detected_class or "—", "Confirmed" if r.decision == "accepted" else "Rejected",
                    f"{r.confidence:.0%}" if r.confidence else "—", ctx.get("frame", r.frame_id.split(":")[-2] if ":" in r.frame_id else "—"),
                    ctx.get("source") or (r.frame_id.split(":")[0] if ":" in r.frame_id else "—")]
            for c, v in enumerate(vals):
                self.table.setItem(row, c, QTableWidgetItem(str(v)))
        self._render_rules()

    # ------------------------------------------------------------------ rules
    def _render_rules(self) -> None:
        svc = self.app.service
        clear_layout(self.rules_body)
        data = svc.rule_proposals()
        ev, items = data["evidence"], data["items"]
        active = svc.rules.active()

        head = QHBoxLayout()
        head.addWidget(label(f"Active rule version: v{active.version}", ""))
        head.addWidget(Pill("built-in policy" if not active.class_auto_redact_confidence else "custom overrides",
                            "neutral" if not active.class_auto_redact_confidence else "accent"))
        head.addStretch()
        w = QWidget()
        w.setLayout(head)
        self.rules_body.addWidget(w)
        if active.class_auto_redact_confidence:
            self.rules_body.addWidget(label("Overrides: " + ", ".join(f"{k} auto-redact ≥ {v:.2f}" for k, v in active.class_auto_redact_confidence.items()), "Muted"))
        else:
            self.rules_body.addWidget(label("Policy: ≥ 80% redacted automatically · 50–80% sent to review · < 50% not redacted on its own.", "Muted"))

        usable = ev.total_records - ev.unusable_records
        self.rules_body.addWidget(label(f"Evidence: {usable} usable decision(s) of {ev.total_records} logged.", "Muted"))
        for n in ev.notes:
            b = Banner()
            b.show_message(n, "warn")
            self.rules_body.addWidget(b)

        if not items:
            self.rules_body.addWidget(label("No proposals. A proposal needs at least 10 reviewed detections for a class in the 50–80% band, "
                                            "of which ≥ 95% were confirmed.", "Faint", wrap=True))
        for entry in items:
            p, rep = entry["proposal"], entry["validation"]
            card = Card(flat=True, padding=14)
            card.add(label(p.summary(), "SectionTitle"))
            card.add(label(p.reason, "Muted", wrap=True))
            for chk in rep.checks:
                line = QHBoxLayout()
                line.addWidget(Pill("pass" if chk["passed"] else "fail", "ok" if chk["passed"] else "danger"))
                line.addWidget(label(f"{chk['name']}: {chk['detail']}", "Faint", wrap=True), 1)
                holder = QWidget()
                holder.setLayout(line)
                card.add(holder)
            row = QHBoxLayout()
            approve = button("Approve and activate", "Primary", callback=lambda p=p, r=rep: self._approve(p, r))
            approve.setEnabled(rep.passed)
            row.addWidget(approve)
            row.addWidget(button("Reject proposal", "", callback=lambda p=p: self._reject(p)))
            row.addStretch()
            card.add(row)
            self.rules_body.addWidget(card)

        hist = svc.rules.history()
        if len(hist) > 1:
            self.rules_body.addWidget(label("Version history", "SectionTitle"))
            for v in reversed(hist[-6:]):
                line = QHBoxLayout()
                line.addWidget(label(f"v{v.version}", ""))
                line.addWidget(label(time.strftime("%Y-%m-%d %H:%M", time.localtime(v.created_at)), "Faint"))
                line.addWidget(label(v.note or "—", "Muted", wrap=True), 1)
                if v.version == active.version:
                    line.addWidget(Pill("active", "ok"))
                else:
                    line.addWidget(button("Roll back to this", "Ghost", callback=lambda n=v.version: self._rollback(n)))
                holder = QWidget()
                holder.setLayout(line)
                self.rules_body.addWidget(holder)
        audit = svc.rules.audit(5)
        if audit:
            self.rules_body.addWidget(label("Audit trail: " + " · ".join(f"{a['action']} v{a['version']}" for a in audit), "Faint", wrap=True))

    def _approve(self, p, rep) -> None:
        if QMessageBox.question(self, "Activate rule", f"{p.summary()}\n\nThis makes more detections redact automatically without review. Continue?") != QMessageBox.Yes:
            return
        try:
            v = self.app.service.approve_proposal(p, rep)
            self.banner.show_message(f"Rule v{v.version} is active. You can roll it back at any time.", "ok")
        except Exception as exc:
            self.banner.show_message(f"Could not activate: {exc}", "danger")
        self.on_show()

    def _reject(self, p) -> None:
        self.app.service.reject_proposal(p)
        self.banner.show_message("Proposal rejected and recorded in the audit trail.", "info")
        self.on_show()

    def _rollback(self, version: int) -> None:
        if QMessageBox.question(self, "Roll back", f"Restore the rules from v{version}? (History is kept.)") != QMessageBox.Yes:
            return
        v = self.app.service.rollback_rules(version)
        self.banner.show_message(f"Rolled back: v{v.version} is now active.", "ok")
        self.on_show()
