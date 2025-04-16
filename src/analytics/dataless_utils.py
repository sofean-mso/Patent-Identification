# Copyright 2025 FIZ-Karlsruhe (Mustafa Sofean)

import numpy as np
from snorkel.labeling import labeling_function, LFAnalysis
from snorkel.labeling.model import LabelModel
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import filter_unlabeled_dataframe
from snorkel.utils import probs_to_preds

# Define the label mappings for convenience
PLASMA_MED = 0
PLASMA_DECO = 1
NO_PLASMA = -1

PLASMA_TECH_SET = {'ionized gas',
                   'ionized plasma',
                   'plasma source',
                   'plasma physics',
                   'non-thermal plasma',
                   'non thermal plasma',
                   'microwave plasma',
                   'plasma microwave',
                   'plasma sterilizer',
                   'sterilization device',
                   'plasma medicine',
                   'plasma etching',
                   'plasma coating',
                   'plasma sterilization',
                   'plasma surgery',
                   'cold atmospheric plasma',
                   'plasma jet',
                   'plasma jets',
                   'plasma torch',
                   'plasma apparatus',
                   'plasma torches',
                   'plasma atomizer',
                   'cold plasma',
                   'jet plasma',
                   'dielectric barrier discharge',
                   'plasma-activated media',
                   'plasma activated media',
                   'plasma-activated water',
                   'plasma activated water',
                   'plasma-activated liquid',
                   'plasma activated liquid',
                   'plasma toothbrush',
                   'plasma cleaning',
                   'plasma generating',
                   'plasma processing'
                   'plasma treatment',
                   'plasma cutting',
                   'plasma arc cutting',
                   'plasma device',
                   'plasma gas',
                   'plasma generator',
                   'plasma generation device',
                   'plasma chamber',
                   'plasma air sterilization',
                   'plasma disinfection',
                   'plasma decontamination',
                   'plasma polymerization',
                   'plasma polymerizing',
                   'low-pressure plasma',
                   'low pressure plasma',
                   'plasma diagnostics',
                   'plasma-enhanced chemical vapor deposition',
                   'plasma enhanced chemical vapor deposition',
                   'plasma deposition'
                   }

PLASMA_MED_SEED_SET = {
    'burn wound',
    'wound healing',
    'cancer treatment',
    'tumor treatment',
    'chronic wound',
    'skin disease',
    'skin care',
    'hair removal',
    'tissue regeneration',
    'tattoo removal',
    'skin treatment',
    'plasma surgery',
    'skin rejuvenation',
}
PLASMA_MED_SEED_SET1 = {'tumor',
                        'burn',
                        'wound',
                        'cancer',
                        'chronic',
                        'skin',
                        'medicine',
                        'teeth',
                        ' gum ',
                        'bacteria',
                        'virus',
                        'hair',
                        'tissue',
                        'tattoo',
                        'skin',
                        'surgery',
                        'infection',
                        'dermatitis',
                        'tissue',
                        'oncology',
                        'fungi',
                        'therapy',
                        'tissue',
                        'disease',
                        'teeth'
                        }

PLASMA_DECO_SEED_LIST = {
    'sanitization',
    'decontamination',
    'contamination',
    'toxin',
    'packaging',
    'disinfecting',
    'purification',
    ' food ',
    'decontaminate',
    'textile',
    'chemical contaminant',
    ' surface',
    ' cleaning '
    'water treatment',
    'decontaminating',
    'sterilization',
    'sterilizing'
}

PLASMA_MED_TECH_SET = {'plasma medicine'
                       }

PLASMA_DECO_TECH_SET = {'plasma decontamination'
                        }


@labeling_function()
def lf_plasma_med(x):
    for seed in PLASMA_MED_SEED_SET1:
        if seed in x.text.lower() and 'plasma' in x.text.lower():
            return PLASMA_MED

    return NO_PLASMA


@labeling_function()
def lf_plasma_med_tech(x):
    for seed in PLASMA_MED_TECH_SET:
        if seed in x.text.lower() and 'plasma' in x.text.lower():
            return PLASMA_MED

    return NO_PLASMA


@labeling_function()
def lf_plasma_deco(x):
    for seed in PLASMA_DECO_SEED_LIST:
        if seed in x.text.lower() and 'plasma' in x.text.lower():
            return PLASMA_DECO

    return NO_PLASMA


@labeling_function()
def lf_plasma_deco_tech(x):
    for seed in PLASMA_DECO_TECH_SET:
        if seed in x.text.lower() and 'plasma' in x.text.lower():
            return PLASMA_DECO

    return NO_PLASMA


def create_training_data(df_train):
    # combining  all labeling functions (LFs)
    lfs = [lf_plasma_med, lf_plasma_deco, lf_plasma_med_tech, lf_plasma_deco_tech]

    # Apply the LFs to the unlabeled training data
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(df_train)

    # Train the label model and compute the training labels
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train, n_epochs=500, lr=0.001, log_freq=50, seed=123)
    df_train["label"] = label_model.predict(L=L_train, tie_break_policy="abstain")

    # Evaluate performance on training set
    # coverage_check_out, coverage_check  = (L_train != NO_PLASMA).mean(axis=0)
    # print(f"check_out coverage: {coverage_check_out * 100:.1f}%")
    # print(f"check coverage: {coverage_check * 100:.1f}%")
    lf_summary = LFAnalysis(L=L_train, lfs=lfs).lf_summary()
    print(lf_summary)

    df_train = df_train[df_train.label != NO_PLASMA]

    return df_train
