
# Copyright 2024 FIZ-Karlsruhe (Mustafa Sofean)

import warnings
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
from snorkel.labeling import labeling_function, LFAnalysis
from snorkel.labeling.model import LabelModel
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import filter_unlabeled_dataframe
from snorkel.utils import probs_to_preds

# Define the label mappings for convenience
INFO_SEC = 0
NON_INF_SEC = 1
ABSTAIN = -1

INFO_SEC_CORE_SET = {
    'protecting computer',
    'information security',
    'network security',
    'cryptography',
    'cryptographic',
    'data integrity',
    'user authentication',
    'data authentication',
    'non-repudiation',
    'non repudiation',
    'ciphertext',
    'encryption',
    'decryption',
    'key management',
    'network security',
    'firewall',
    'user authorization',
    'cryptocurrency security',
    'homomorphic encryption',
    'vulnerability detection',
    'fraud risk',
    'credit card fraud',
    'computer virus'
                   }

INFO_SEC_ENCRYPTION_SET = {
                     'symmetric encryption',
                    'asymmetric encryption',
                    'block cipher',
                    'stream cipher',
                    'advanced encryption standard',
                    'data encryption standard',
                    'rivest-shamir-adleman',
                    'rivest shamir adleman',
                    'elliptic curve cryptography',
                    'diffie-hellman key exchange',
                    'diffie hellman key exchange'
                    'elgamal encryption',
                    'secure hash algorithm',
                    'message digest',
                    'hash function',
                    'hash-based message authentication code',
                    'hash based message authentication code'
                    'digital signature',
                    'digital certificate',
                    'certificate authority',
                    'message digest',
                    'message authentication code',
                    'checksum – basic error-checking technique',
                    'checksum basic error-checking technique',
                    'quantum key'
                    }
INFO_SEC_INFRA_SET = { 'public key infrastructure',
                        'certificate revocation list',
                        'online certificate status protocol',
                        'hardware security module',
                        'cryptographic protocol',
                        'certificate authority',
                        'public key',
                        'private key',
                        'digital signature'
                        }

INFO_SEC_NET_SET = { 'network security',
    'firewalls',
    'intrusion detection system',
    'intrusion prevention system',
    'virtual private network',
    'secure socket layer',
    'network access control',
    'denial of service',
    'ipsec',
    'dns security',
    'port scanning',
    'wireless security',
    'proxy servers',
    'zero trust network',
    'proxy server',
    'network access control',
    'man in the middle attack'

  }

INFO_SEC_APPs_SET = { 'secure software',
    'sql injection',
    'cross site scripting',
    'cross site request forgery',
    'code review',
    'application security testing',
    'threat modeling',
    'secure coding standards',
    'api security',
    'web application firewalls',
    'antivirus',
    'endpoint detection and response',
    'device encryption',
    'endpoint firewalls',
    'application whitelisting',
    'malware sandboxing',
    'secure boot',
    'anti phishing tools'
                       }

INFO_SEC_ACESS_SET = { 'authentication',
    'authorization',
    'multi-factor authentication',
    'multi factor authentication',
    'single sign on',
    'identity federation',
    'role based access control',
    'attribute based access control',
    'privileged access management',
    'identity lifecycle management',
    'credential management',
    'least privilege',
    'just-in-time access',
    'directory services',
    'biometric authentication',
    'session management',
    'identity governance',
    'access certification',
    'password policies',
    'dynamic password',
    'password protection',
    'pin password',
    'user access control',
    'network access control'
                        }


INFO_SEC_CLOUD_SET = { 'cloud access security broker',
    'cloud identity',
    'cloud encryption',
    'secure cloud',
    'cloud security',
    'cloud-native firewalls',
    'data loss prevention',
    'cloud compliance',
    'virtual private cloud',
    'container security',
    'kubernetes security',
    'serverless security',
    'saas security',
    'cloud threat intelligence'
                         }
INFO_SEC_GENERAL_SET = {
"blockchains",
    "communication privacy",
    "computer security",
    "controlling access to content",
    "digital rights management",
    "fault induction attack",
    "identity verification",
    "information technology security",
    "license key",
    "malware",
    "malicious",
    "malicious software",
    "mobile payment",
    "mobile security",
    "secure communication",
    "secure network",
    "securing remote access",
    "security protection",
    "server security",
    "software licensing",
    "trojan horse",
    "url attack",
    "user credentials",
    "web attack",
    "web security"
}

SEURITY_TERMS = [
'securely',
    'security',
    'attack',
    'signature',
    'secure',
    'risks',
    'attacks',
    'untrusted',
    'trusted',
    'privacy',
    'protection',
    'authenticating',
    'authentication',
    'authorization',
    'access control',
    'secret',
    'authorizing',
    'protected',
    'validating',
    'validation',
    'licensing',
    'license',
    'encrypt',
    'blockchain',
    'authenticate',
    'trust',
    'confidential',
    'securing',
    'confidentiality',
    'plaintext', "cryptography",
    "anit virus",
    "anti theft",
    "user access",
    "protecting data",
    "screen lock",
    "session key",
    "identity management",
    "risks",
    "copyright",
    "sensitive information",
    "key generation",
    "account theft",
    "cyber",
    "verification",
    "abnormal request",
    "decrypt",
    "encrypt",
    "user account",
    "password",
    "safety",
    "user identification",
    "rights",
    "users to access",
    "unlock",
    "identification",
    "access method",
    "digital asset",
    "data access",
    "enciphering",
    "prevent leakage",
    "anonymous access",
    "login",
    "user to access",
    "eavesdropping",
    "authorisation",
    "authorising",
    "cipher key",
    "vulnerabilities",
    "vulnerability",
    "authorized",
    "anonymization",
    "data leak",
    "unlocking",
    "accessing network",
    "nft token",
    "fingerprint",
    "gateway",
    "cipher",
    "fraud risk"
]

ALL_INFO_SEC_SET = INFO_SEC_CORE_SET | INFO_SEC_ENCRYPTION_SET | INFO_SEC_INFRA_SET | INFO_SEC_NET_SET | INFO_SEC_APPs_SET | INFO_SEC_ACESS_SET | INFO_SEC_CLOUD_SET | INFO_SEC_GENERAL_SET

@labeling_function()
def lf_info_sec_core_tech(x):
    for tech in INFO_SEC_CORE_SET:
        if tech in x.text.lower():
            return INFO_SEC

    return ABSTAIN


@labeling_function()
def lf_info_sec_encryption(x):
    for seed in INFO_SEC_ENCRYPTION_SET:
        if seed in x.text.lower():
            return INFO_SEC

    return ABSTAIN


@labeling_function()
def lf_info_sec_INFRA(x):
    for seed in INFO_SEC_INFRA_SET:
        if seed in x.text.lower():
            return INFO_SEC

    return ABSTAIN


@labeling_function()
def lf_info_sec_net(x):
    for seed in INFO_SEC_NET_SET:
        if seed in x.text.lower():
            return INFO_SEC

    return ABSTAIN


@labeling_function()
def lf_info_sec_apps(x):
    for seed in INFO_SEC_APPs_SET:
        if seed in x.text.lower():
            return INFO_SEC

    return ABSTAIN

@labeling_function()
def lf_info_sec_access(x):
    for seed in INFO_SEC_ACESS_SET:
        if seed in x.text.lower():
            return INFO_SEC

    return ABSTAIN

@labeling_function()
def lf_info_sec_cloud(x):
    for seed in INFO_SEC_CLOUD_SET:
        if seed in x.text.lower():
            return INFO_SEC

    return ABSTAIN

@labeling_function()
def lf_info_sec_core_general(x):
    for tech in INFO_SEC_GENERAL_SET:
        if tech in x.text.lower():
            return INFO_SEC

    return ABSTAIN

@labeling_function()
def lf_no_info_sec(x):
    no_info_sec = False
    for seed in ALL_INFO_SEC_SET:
        if seed in x.text.lower():
            no_info_sec = True

    for seed in SEURITY_TERMS:
        if seed in x.text.lower():
            no_info_sec = True

    if no_info_sec is False:
        return NON_INF_SEC

    return ABSTAIN



def create_info_sec_training_dataset(df_train):
    # combining  all labeling functions (LFs)
    lfs = [lf_info_sec_core_tech,
           lf_info_sec_encryption,
           lf_info_sec_INFRA,
           lf_info_sec_net,
           lf_info_sec_apps,
           lf_info_sec_access,
           lf_info_sec_cloud,
           lf_info_sec_core_general,
           lf_no_info_sec
           ]

    # Apply the LFs to the unlabeled training data
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(df_train)

    # Train the label model and compute the training labels
    label_model = LabelModel(cardinality=3, verbose=True)
    label_model.fit(L_train, n_epochs=500, lr=0.001, log_freq=50, seed=123)
    df_train["label"] = label_model.predict(L=L_train, tie_break_policy="abstain")

    # Evaluate performance on training set
    # coverage_check_out, coverage_check  = (L_train != NO_PLASMA).mean(axis=0)
    # print(f"check_out coverage: {coverage_check_out * 100:.1f}%")
    # print(f"check coverage: {coverage_check * 100:.1f}%")
    lf_summary = LFAnalysis(L=L_train, lfs=lfs).lf_summary()
    print(lf_summary)

    #df_train = df_train[df_train.label != NO_PLASMA]

    return df_train
