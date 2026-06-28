# Анализ ошибок между датасетами

## Что сравнивать

Для защиты удобно показывать не все 198 классов SD-198, а группы заболеваний, которые пересекаются со старыми датасетами. В таблице `overlap_quality_comparison.csv` сравниваются recall/F1 на HAM10000, Skin Extra и агрегированное качество на соответствующих классах SD-198.

Такой формат уменьшает объем таблицы: устойчивые классы можно описать обобщенно, а подробно разобрать только группы, где качество снизилось или модель часто путает диагнозы.

## Пересекающиеся заболевания

- Actinic keratosis: HAM10000 recall=0.612, Skin Extra recall=0.600, SD-198 grouped recall=0.333, support в SD-198=9.
- Melanoma: HAM10000 recall=0.725, Skin Extra recall=0.600, SD-198 grouped recall=0.529, support в SD-198=17.
- Basal cell carcinoma: HAM10000 recall=0.883, Skin Extra recall=нет, SD-198 grouped recall=0.667, support в SD-198=9.
- Dermatofibroma: HAM10000 recall=0.647, Skin Extra recall=0.900, SD-198 grouped recall=0.667, support в SD-198=9.
- Vascular lesion: HAM10000 recall=0.909, Skin Extra recall=1.000, SD-198 grouped recall=0.667, support в SD-198=21.
- Tinea Ringworm Candidiasis: HAM10000 recall=нет, Skin Extra recall=1.000, SD-198 grouped recall=0.684, support в SD-198=57.
- Benign keratosis: HAM10000 recall=0.655, Skin Extra recall=1.000, SD-198 grouped recall=0.722, support в SD-198=18.
- Atopic Dermatitis: HAM10000 recall=нет, Skin Extra recall=1.000, SD-198 grouped recall=0.750, support в SD-198=4.
- Melanocytic nevus: HAM10000 recall=0.882, Skin Extra recall=0.900, SD-198 grouped recall=0.812, support в SD-198=64.
- Squamous cell carcinoma: HAM10000 recall=нет, Skin Extra recall=0.500, SD-198 grouped recall=нет, support в SD-198=0.

## Самые заметные ошибки SD-198

- Compound_Nevus -> Blue_Nevus: 4 из 9 (44.4%), внутри категории (pigmented_lesions -> pigmented_lesions).
- Dysplastic_Nevus -> Nevus_Incipiens: 3 из 9 (33.3%), внутри категории (pigmented_lesions -> pigmented_lesions).
- Tinea_Corporis -> Nummular_Eczema: 3 из 9 (33.3%), между категориями (fungal -> eczema_dermatitis).
- Angioma -> Blue_Nevus: 2 из 9 (22.2%), между категориями (vascular -> pigmented_lesions).
- Basal_Cell_Carcinoma -> Bowen's_Disease: 2 из 9 (22.2%), внутри категории (tumor -> tumor).
- Cellulitis -> Stasis_Ulcer: 2 из 5 (40.0%), внутри категории (other -> other).
- Cutaneous_T-Cell_Lymphoma -> Drug_Eruption: 2 из 3 (66.7%), внутри категории (other -> other).
- Darier-White_Disease -> Steroid_Acne: 2 из 5 (40.0%), внутри категории (other -> other).
- Dry_Skin_Eczema -> Erythema_Craquele: 2 из 8 (25.0%), между категориями (eczema_dermatitis -> other).
- Dyshidrosiform_Eczema -> Tinea_Manus: 2 из 9 (22.2%), между категориями (eczema_dermatitis -> fungal).
- Dysplastic_Nevus -> Junction_Nevus: 2 из 9 (22.2%), внутри категории (pigmented_lesions -> pigmented_lesions).
- Epidermoid_Cyst -> Neurofibroma: 2 из 9 (22.2%), внутри категории (other -> other).

## Ошибки между разными категориями

- Tinea_Corporis -> Nummular_Eczema: 3 из 9 (33.3%) (fungal -> eczema_dermatitis).
- Angioma -> Blue_Nevus: 2 из 9 (22.2%) (vascular -> pigmented_lesions).
- Dry_Skin_Eczema -> Erythema_Craquele: 2 из 8 (25.0%) (eczema_dermatitis -> other).
- Dyshidrosiform_Eczema -> Tinea_Manus: 2 из 9 (22.2%) (eczema_dermatitis -> fungal).
- Lymphomatoid_Papulosis -> Halo_Nevus: 2 из 3 (66.7%) (other -> pigmented_lesions).
- Onychoschizia -> Beau's_Lines: 2 из 4 (50.0%) (nail -> other).
- Pityriasis_Rosea -> Guttate_Psoriasis: 2 из 5 (40.0%) (other -> psoriasis).
- Seborrheic_Dermatitis -> Angular_Cheilitis: 2 из 9 (22.2%) (eczema_dermatitis -> other).
- Seborrheic_Dermatitis -> Rosacea: 2 из 9 (22.2%) (eczema_dermatitis -> other).
- Stasis_Edema -> Stasis_Dermatitis: 2 из 9 (22.2%) (other -> eczema_dermatitis).

## Устойчиво распознаваемые классы

Классов с precision >= 0.8, recall >= 0.9 и support >= 5: 5. Их можно описать обобщенно как устойчиво распознаваемые, а не выносить в большую таблицу.

## Как это можно описать в работе

Наиболее информативно разделить ошибки на два типа. Первый тип - ошибки внутри одной клинической категории, например между разновидностями невусов; такие ошибки показывают, что модель улавливает общий тип поражения, но недостаточно точно различает близкие подтипы. Второй тип - ошибки между категориями, например грибковые поражения, ошибочно отнесенные к экземам/дерматитам; такие случаи более критичны и требуют отдельного обсуждения.

Для пересекающихся классов видно, сохраняется ли качество при переходе от старых датасетов к SD-198. Если recall на группе SD-198 ниже, это можно связать с большей детализацией классов, меньшим числом примеров на класс и визуальной похожестью заболеваний.

Сгенерированные файлы:

- overlap_quality_comparison.csv
- sd198_top_confusions_grouped.csv
- sd198_category_confusion_summary.csv
- sd198_stable_high_quality_classes.csv
