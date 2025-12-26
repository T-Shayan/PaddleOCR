# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import mlflow
from rapidfuzz.distance import Levenshtein
from difflib import SequenceMatcher

import numpy as np
import string
from .bleu import compute_bleu_score, compute_edit_distance


def normalize_arabic_to_persian(text: str) -> str:
    """
    Normalize Arabic text to Persian (Farsi) standard.
    Converts Kaf, Yeh, Digits, and removes diacritics.
    """
    if not text:
        return ""

    # Using a translation table for O(n) performance
    translation_table = str.maketrans({
        "ك": "ک",  # Arabic Kaf to Persian
        "ي": "ی",  # Arabic Yeh (with dots) to Persian
        "ى": "ی",  # Alif Maksura to Persian Yeh
        "ة": "ه",  # Teh Marbuta to Heh
        "أ": "ا",  # Alef with Hamza Above to Alef
        "إ": "ا",  # Alef with Hamza Below to Alef
        "آ": "ا",  # Alef with Madda to Alef
        "ٱ": "ا",  # Alef Wasla to Alef
        "ؤ": "و",  # Waw with Hamza to Waw
        "ئ": "ی",  # Yeh with Hamza to Persian Yeh

        # Arabic Digits
        "٠": "۰", 
        "١": "۱", 
        "٢": "۲", 
        "٣": "۳", 
        "٤": "۴",
        "٥": "۵", 
        "٦": "۶", 
        "٧": "۷", 
        "٨": "۸", 
        "٩": "۹",
    })
    
    text = text.strip().translate(translation_table)

    return text


class RecMetric(object):
    def __init__(
        self, main_indicator="acc", is_filter=False, ignore_space=True, **kwargs
    ):
        self.main_indicator = main_indicator
        self.is_filter = is_filter
        self.ignore_space = ignore_space
        self.eps = 1e-5
        self.reset()

    def _normalize_text(self, text):
        text = "".join(
            filter(lambda x: x in (string.digits + string.ascii_letters), text)
        )
        return text.lower()

    def __call__(self, pred_label, *args, **kwargs):
        preds, labels = pred_label

        correct_num = 0
        all_num = 0

        norm_edit_dis = 0.0
        char_edit_dist = 0
        char_len = 0

        char_edit_dist_ic = 0
        char_edit_dist_ichar = 0

        word_edit_dist = 0
        word_len = 0

        conf_sum = 0.0

        for (pred, pred_conf), (target, _) in zip(preds, labels):
            if self.ignore_space:
                pred = pred.replace(" ", "")
                target = target.replace(" ", "")

            if self.is_filter:
                pred = self._normalize_text(pred)
                target = self._normalize_text(target)

            # Line-level
            if pred == target:
                correct_num += 1
            all_num += 1

            # Normalized edit distance (line)
            norm_edit_dis += Levenshtein.normalized_distance(pred, target)

            # Character-level (CER)
            char_edit_dist += Levenshtein.distance(pred, target)
            char_len += len(target)

            # Character-level insensitive case (CER)
            char_edit_dist_ic += Levenshtein.distance(
                pred.lower(), target.lower()
            )

            # Character-level insensitive characters (CER)
            char_edit_dist_ichar += Levenshtein.distance(
                normalize_arabic_to_persian(pred.lower()), 
                normalize_arabic_to_persian(target.lower())
            )

            # Word-level (WER)
            pred_words = pred.split()
            target_words = target.split()
            word_edit_dist += Levenshtein.distance(pred_words, target_words)
            word_len += len(target_words)

            # Confidence (optional)
            conf_sum += float(pred_conf)

        self.correct_num += correct_num
        self.all_num += all_num
        self.char_len += char_len
        self.word_len += word_len
        self.conf_sum += conf_sum
        self.norm_edit_dis += norm_edit_dis
        self.char_edit_dist += char_edit_dist
        self.char_edit_dist_ic += char_edit_dist_ic
        self.char_edit_dist_ichar += char_edit_dist_ichar
        self.word_edit_dist += word_edit_dist
        self.list_of_pairs.append((preds[0][0], labels[0][0]))

        cer = char_edit_dist / (char_len + self.eps)
        wer = word_edit_dist / (word_len + self.eps)
        cer_ic = char_edit_dist_ic / (char_len + self.eps)
        cer_ichar = char_edit_dist_ichar / (char_len + self.eps)

        return {
            "acc": correct_num / (all_num + self.eps),
            "norm_edit_dis": 1 - norm_edit_dis / (all_num + self.eps),
            "cer": cer,
            "wer": wer,
            "cer_icase": cer_ic,
            "cer_ichar": cer_ichar,
            "avg_conf": conf_sum / (all_num + self.eps),
        }

    def get_metric(self, step=None):
        """
        return metrics {
                 'acc': 0,
                 'norm_edit_dis': 0,
            }
        """
        if step is not None:
            rows = []
            for pred, label in self.list_of_pairs:
                rows.append({
                    "step": step,
                    "pred": pred,
                    "label": label,
                })

            df = pd.DataFrame(rows)
            mlflow.log_table(df, f"evaluation/preds_labels.json")

        acc = 1.0 * self.correct_num / (self.all_num + self.eps)
        norm_edit_dis = 1 - self.norm_edit_dis / (self.all_num + self.eps)
        cer = self.char_edit_dist / (self.char_len + self.eps)
        wer = self.word_edit_dist / (self.word_len + self.eps)
        cer_ic = self.char_edit_dist_ic / (self.char_len + self.eps)
        cer_ichar = self.char_edit_dist_ichar / (self.char_len + self.eps)
        avg_conf = self.conf_sum / (self.all_num + self.eps)
        self.reset()
        return {
            "acc": acc, 
            "cer": cer,
            "wer": wer,
            "cer_icase": cer_ic,
            "cer_ichar": cer_ichar,
            "norm_edit_dis": norm_edit_dis,
            "avg_conf": avg_conf,
        }

    def reset(self):
        self.correct_num = 0
        self.all_num = 0
        self.norm_edit_dis = 0
        self.char_edit_dist = 0
        self.char_edit_dist_ic = 0
        self.char_edit_dist_ichar = 0
        self.word_edit_dist = 0
        self.word_len = 0
        self.char_len = 0
        self.conf_sum = 0
        self.list_of_pairs = []

class CNTMetric(object):
    def __init__(self, main_indicator="acc", **kwargs):
        self.main_indicator = main_indicator
        self.eps = 1e-5
        self.reset()

    def __call__(self, pred_label, *args, **kwargs):
        preds, labels = pred_label
        correct_num = 0
        all_num = 0
        for pred, target in zip(preds, labels):
            if pred == target:
                correct_num += 1
            all_num += 1
        self.correct_num += correct_num
        self.all_num += all_num
        return {
            "acc": correct_num / (all_num + self.eps),
        }

    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
            }
        """
        acc = 1.0 * self.correct_num / (self.all_num + self.eps)
        self.reset()
        return {"acc": acc}

    def reset(self):
        self.correct_num = 0
        self.all_num = 0


class CANMetric(object):
    def __init__(self, main_indicator="exp_rate", **kwargs):
        self.main_indicator = main_indicator
        self.word_right = []
        self.exp_right = []
        self.word_total_length = 0
        self.exp_total_num = 0
        self.word_rate = 0
        self.exp_rate = 0
        self.reset()
        self.epoch_reset()

    def __call__(self, preds, batch, **kwargs):
        for k, v in kwargs.items():
            epoch_reset = v
            if epoch_reset:
                self.epoch_reset()
        word_probs = preds
        word_label, word_label_mask = batch
        line_right = 0
        if word_probs is not None:
            word_pred = word_probs.argmax(2)
        word_pred = word_pred.cpu().detach().numpy()
        word_scores = [
            SequenceMatcher(
                None, s1[: int(np.sum(s3))], s2[: int(np.sum(s3))], autojunk=False
            ).ratio()
            * (len(s1[: int(np.sum(s3))]) + len(s2[: int(np.sum(s3))]))
            / len(s1[: int(np.sum(s3))])
            / 2
            for s1, s2, s3 in zip(word_label, word_pred, word_label_mask)
        ]
        batch_size = len(word_scores)
        for i in range(batch_size):
            if word_scores[i] == 1:
                line_right += 1
        self.word_rate = np.mean(word_scores)  # float
        self.exp_rate = line_right / batch_size  # float
        exp_length, word_length = word_label.shape[:2]
        self.word_right.append(self.word_rate * word_length)
        self.exp_right.append(self.exp_rate * exp_length)
        self.word_total_length = self.word_total_length + word_length
        self.exp_total_num = self.exp_total_num + exp_length

    def get_metric(self):
        """
        return {
            'word_rate': 0,
            "exp_rate": 0,
        }
        """
        cur_word_rate = sum(self.word_right) / self.word_total_length
        cur_exp_rate = sum(self.exp_right) / self.exp_total_num
        self.reset()
        return {"word_rate": cur_word_rate, "exp_rate": cur_exp_rate}

    def reset(self):
        self.word_rate = 0
        self.exp_rate = 0

    def epoch_reset(self):
        self.word_right = []
        self.exp_right = []
        self.word_total_length = 0
        self.exp_total_num = 0


class LaTeXOCRMetric(object):
    def __init__(self, main_indicator="exp_rate", cal_bleu_score=False, **kwargs):
        self.main_indicator = main_indicator
        self.cal_bleu_score = cal_bleu_score
        self.edit_right = []
        self.exp_right = []
        self.bleu_right = []
        self.e1_right = []
        self.e2_right = []
        self.e3_right = []
        self.exp_total_num = 0
        self.edit_dist = 0
        self.exp_rate = 0
        if self.cal_bleu_score:
            self.bleu_score = 0
        self.e1 = 0
        self.e2 = 0
        self.e3 = 0
        self.reset()
        self.epoch_reset()

    def __call__(self, preds, batch, **kwargs):
        for k, v in kwargs.items():
            epoch_reset = v
            if epoch_reset:
                self.epoch_reset()
        word_pred = preds
        word_label = batch
        line_right, e1, e2, e3 = 0, 0, 0, 0
        bleu_list, lev_dist = [], []
        for labels, prediction in zip(word_label, word_pred):
            if prediction == labels:
                line_right += 1
            distance = compute_edit_distance(prediction, labels)
            bleu_list.append(compute_bleu_score([prediction], [labels]))
            lev_dist.append(Levenshtein.normalized_distance(prediction, labels))
            if distance <= 1:
                e1 += 1
            if distance <= 2:
                e2 += 1
            if distance <= 3:
                e3 += 1

        batch_size = len(lev_dist)

        self.edit_dist = sum(lev_dist)  # float
        self.exp_rate = line_right  # float
        if self.cal_bleu_score:
            self.bleu_score = sum(bleu_list)
            self.bleu_right.append(self.bleu_score)
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3
        exp_length = len(word_label)
        self.edit_right.append(self.edit_dist)
        self.exp_right.append(self.exp_rate)
        self.e1_right.append(self.e1)
        self.e2_right.append(self.e2)
        self.e3_right.append(self.e3)
        self.exp_total_num = self.exp_total_num + exp_length

    def get_metric(self):
        """
        return {
            'edit distance': 0,
            "bleu_score": 0,
            "exp_rate": 0,
        }
        """
        cur_edit_distance = sum(self.edit_right) / self.exp_total_num
        cur_exp_rate = sum(self.exp_right) / self.exp_total_num
        if self.cal_bleu_score:
            cur_bleu_score = sum(self.bleu_right) / self.exp_total_num
        cur_exp_1 = sum(self.e1_right) / self.exp_total_num
        cur_exp_2 = sum(self.e2_right) / self.exp_total_num
        cur_exp_3 = sum(self.e3_right) / self.exp_total_num
        self.reset()
        if self.cal_bleu_score:
            return {
                "bleu_score": cur_bleu_score,
                "edit distance": cur_edit_distance,
                "exp_rate": cur_exp_rate,
                "exp_rate<=1 ": cur_exp_1,
                "exp_rate<=2 ": cur_exp_2,
                "exp_rate<=3 ": cur_exp_3,
            }
        else:

            return {
                "edit distance": cur_edit_distance,
                "exp_rate": cur_exp_rate,
                "exp_rate<=1 ": cur_exp_1,
                "exp_rate<=2 ": cur_exp_2,
                "exp_rate<=3 ": cur_exp_3,
            }

    def reset(self):
        self.edit_dist = 0
        self.exp_rate = 0
        if self.cal_bleu_score:
            self.bleu_score = 0
        self.e1 = 0
        self.e2 = 0
        self.e3 = 0

    def epoch_reset(self):
        self.edit_right = []
        self.exp_right = []
        if self.cal_bleu_score:
            self.bleu_right = []
        self.e1_right = []
        self.e2_right = []
        self.e3_right = []
        self.editdistance_total_length = 0
        self.exp_total_num = 0
