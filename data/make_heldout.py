#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_heldout.py - held-out synthetic evaluation set for PII redaction in
Indian clinical notes (English, Hindi/Devanagari, Hinglish/code-switched).

Stdlib only, deterministic (seed 20261115). Writes heldout.jsonl next to this
script. Everything is synthetic and template-generated; the author did not see
the detector code. No real personal data.

Run:  python3 make_heldout.py
"""
import json
import os
import random
import re
import sys
from collections import Counter
from datetime import date, timedelta

SEED = 20261115
R = random.Random(SEED)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "heldout.jsonl")

N_EN, N_HI, N_MX = 120, 50, 30


def ch(seq):
    return R.choice(seq)


def opt(p):
    return R.random() < p


def ri(a, b):
    return R.randint(a, b)


DIG = "0123456789"
DEVA_TR = str.maketrans("0123456789", "०१२३४५६७८९")
LAT_TR = str.maketrans("०१२३४५६७८९", "0123456789")


def deva(s):
    return s.translate(DEVA_TR)


class Raw(str):
    """Plain text that must keep Latin digits even in a Devanagari-digit note."""


# --------------------------------------------------------------------------
# Verhoeff (Aadhaar check digit)
# --------------------------------------------------------------------------
V_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
V_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]
V_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_digit(num):
    c = 0
    for i, chd in enumerate(reversed(num)):
        c = V_D[c][V_P[(i + 1) % 8][int(chd)]]
    return V_INV[c]


def verhoeff_ok(num):
    c = 0
    for i, chd in enumerate(reversed(num)):
        c = V_D[c][V_P[i % 8][int(chd)]]
    return c == 0


assert verhoeff_digit("236") == 3 and verhoeff_ok("2363") and not verhoeff_ok("2364")
assert verhoeff_digit("12345") == 1 and verhoeff_ok("123451")


def digits(n, first=DIG):
    return ch(first) + "".join(ch(DIG) for _ in range(n - 1))


def not_verhoeff(n, first=DIG):
    while True:
        s = digits(n, first)
        if not verhoeff_ok(s):
            return s


# --------------------------------------------------------------------------
# Document builder with exact offsets
# --------------------------------------------------------------------------
class Doc:
    NOCASE = {"EMAIL", "ABHA_ADDRESS"}
    NODEVA = {"EMAIL", "ABHA_ADDRESS", "ABHA_NUMBER", "PAN", "VOTER_ID", "MRN"}

    def __init__(self):
        self.buf, self.n, self.ents = [], 0, []
        self.case = None   # None | 'upper' | 'lower'
        self.deva = False  # convert digits to Devanagari

    def _x(self, s, typ=None):
        raw = isinstance(s, Raw)
        s = str(s)
        if self.case and typ not in self.NOCASE:
            s = s.upper() if self.case == "upper" else s.lower()
        if self.deva and not raw and typ not in self.NODEVA:
            s = deva(s)
        return s

    def t(self, *items):
        for it in items:
            if it is None or it == "":
                continue
            if isinstance(it, tuple):
                self.e(*it)
            elif isinstance(it, list):
                self.t(*it)
            else:
                s = self._x(it)
                self.buf.append(s)
                self.n += len(s)
        return self

    def e(self, s, typ):
        s = self._x(s, typ)
        st = self.n
        self.buf.append(s)
        self.n += len(s)
        self.ents.append({"start": st, "end": self.n, "type": typ, "text": s})
        return self

    def text(self):
        return "".join(self.buf)


def E(s, typ):
    return (s, typ)


def kv(d, fields, sep=None, pair=0.3):
    sep = sep if sep is not None else ch([": ", ": ", ": ", " : ", " - ", ":- ", ":", " "])
    i = 0
    while i < len(fields):
        lab, val = fields[i]
        d.t(lab, sep, val)
        if i + 1 < len(fields) and opt(pair):
            lab2, val2 = fields[i + 1]
            d.t(ch(["    ", "\t", "  |  ", "   "]), lab2, sep, val2)
            i += 1
        d.t("\n")
        i += 1


# --------------------------------------------------------------------------
# Dates and numbers
# --------------------------------------------------------------------------
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH = ["January", "February", "March", "April", "May", "June", "July", "August",
         "September", "October", "November", "December"]
HI_MONTH = ["जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर",
            "अक्टूबर", "नवंबर", "दिसंबर"]


def fdate(dt, style=None):
    style = style or ch(["dmy/", "dmy/", "dmy-", "dmy.", "d b Y", "d-b-Y", "B d, Y", "ymd", "dmy/y"])
    d, m, y = dt.day, dt.month, dt.year
    if style == "dmy/":
        return f"{d:02d}/{m:02d}/{y}"
    if style == "dmy-":
        return f"{d:02d}-{m:02d}-{y}"
    if style == "dmy.":
        return f"{d:02d}.{m:02d}.{y}"
    if style == "d b Y":
        return f"{d} {MON[m-1]} {y}"
    if style == "d-b-Y":
        return f"{d:02d}-{MON[m-1]}-{y}"
    if style == "B d, Y":
        return f"{MONTH[m-1]} {d}, {y}"
    if style == "ymd":
        return f"{y}-{m:02d}-{d:02d}"
    if style == "dmy/y":
        return f"{d:02d}/{m:02d}/{y % 100:02d}"
    if style == "hi":
        return f"{d} {HI_MONTH[m-1]} {y}"
    raise ValueError(style)


def rand_date(year=2026, m_lo=1, m_hi=10):
    start = date(year, m_lo, 1)
    end = date(year, m_hi, 28)
    return start + timedelta(days=ri(0, (end - start).days))


def inr(n):
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts) + "," + tail


# --------------------------------------------------------------------------
# Identifier generators
# --------------------------------------------------------------------------
def aadhaar_raw():
    body = digits(11, "23456789")
    return body + str(verhoeff_digit(body))


def fmt_aadhaar(style=None, dev=False):
    num = aadhaar_raw()
    style = style or ch(["space", "space", "space", "plain", "dash", "mask"])
    if style == "space":
        s = f"{num[:4]} {num[4:8]} {num[8:]}"
    elif style == "plain":
        s = num
    elif style == "dash":
        s = f"{num[:4]}-{num[4:8]}-{num[8:]}"
    else:
        s = ch([f"XXXX XXXX {num[8:]}", f"xxxx-xxxx-{num[8:]}", f"********{num[8:]}",
                f"XXXX-XXXX-{num[8:]}", f"XXXXXXXX{num[8:]}"])
    return deva(s) if dev else s


def abha_number(style=None):
    first = "91" if opt(0.6) else ch("123456789") + ch(DIG)
    n = first + digits(12)
    style = style or ch(["dash"] * 6 + ["space", "space", "plain"])
    if style == "dash":
        return f"{n[:2]}-{n[2:6]}-{n[6:10]}-{n[10:]}"
    if style == "space":
        return f"{n[:2]} {n[2:6]} {n[6:10]} {n[10:]}"
    return n


def pan_no(p):
    initial = (p.rom_last or p.rom_first or "x")[0].upper()
    if not initial.isalpha():
        initial = "K"
    L = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "".join(ch(L) for _ in range(3)) + "P" + initial + digits(4) + ch(L)


def epic_no():
    L = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "".join(ch(L) for _ in range(3)) + digits(7)


def mobile_raw():
    return ch("9999888776") + digits(9)


def fmt_mobile(style=None, dev=False):
    m = mobile_raw()
    style = style or ch(["p91s", "p91d", "p91", "zero", "plain", "plain", "plain", "dash", "split", "p91sp"])
    s = {
        "p91s": f"+91 {m[:5]} {m[5:]}",
        "p91d": f"+91-{m}",
        "p91": f"+91{m}",
        "p91sp": f"+91 {m}",
        "zero": f"0{m}",
        "plain": m,
        "dash": f"{m[:5]}-{m[5:]}",
        "split": f"{m[:5]} {m[5:]}",
    }[style]
    return deva(s) if dev else s


STD = [("0522", 7), ("0141", 7), ("080", 8), ("033", 8), ("0484", 7), ("0612", 7), ("0755", 7),
       ("0651", 7), ("011", 8), ("044", 8), ("0361", 7), ("0364", 7), ("05498", 6), ("06254", 6),
       ("0291", 7), ("0172", 7)]


def landline():
    code, n = ch(STD)
    num = ch("2345") + digits(n - 1)
    return ch([f"{code}-{num}", f"{code}-{num}", f"{code} {num}", f"({code}) {num}"])


def phone_en():
    """patient/attendant phone: mostly mobile, sometimes a landline with STD code"""
    return landline() if opt(0.2) else fmt_mobile()


def email_for(p):
    f = re.sub(r"[^a-z]", "", p.rom_first.lower()) or "user"
    l = re.sub(r"[^a-z]", "", (p.rom_last or "").lower())
    y = str(ri(70, 99)) if opt(0.5) else str(ri(1, 999))
    local = ch([f"{f}.{l}" if l else f"{f}{y}", f"{f}{l}{y}", f"{f}_{l}" if l else f"{f}_{y}",
                f"{f[0]}{l}{y}" if l else f"{f}{y}", f"{f}{y}"])
    dom = ch(["gmail.com", "gmail.com", "gmail.com", "yahoo.co.in", "rediffmail.com", "outlook.com",
              "hotmail.com", "yahoo.com"])
    return f"{local}@{dom}"


def abha_address(p, year=None):
    f = re.sub(r"[^a-z]", "", p.rom_first.lower()) or "user"
    l = re.sub(r"[^a-z]", "", (p.rom_last or "").lower())
    y = str(year or ri(1960, 2005))
    local = ch([f"{f}.{y}", f"{f}{l}", f"{f}.{l}", f"{f}_{y}", f"{f}{l}{y[-2:]}", f"{f}{ri(10, 9999)}"])
    return local + ("@sbx" if opt(0.12) else "@abdm")


# hospital / programme identifiers (gold: MRN)
def uhid():
    return ch([f"UH{ri(2019, 2026)}{digits(6)}", f"{ch(['MH', 'DH', 'GMC', 'RIMS', 'SGH'])}/{ri(19, 26)}/{digits(6)}",
               f"{ch('12345')}{digits(9)}", f"UHID{digits(8)}", f"{ri(2020, 2026)}/{digits(6)}"])


def ipd_no():
    return ch([f"IP/{ri(24, 26)}/{digits(6)}", f"{ri(24, 26)}-{digits(5)}", f"IPD-{ri(2401, 2611)}-{digits(4)}",
               f"{digits(5, '123456')}/{ri(24, 26)}", f"IP{digits(7)}"])


def opd_no():
    return ch([f"OPD/{ri(2025, 2026)}/{digits(5)}", f"{digits(4)}/{ri(25, 26)}", f"OP-{digits(7)}",
               f"{ri(2025, 2026)}{digits(7)}", f"R-{digits(6)}"])


def cr_no():
    return not_verhoeff(12, "2")


def mrd_no():
    return ch([f"MRD-{digits(7)}", f"{digits(6, '123456')}", f"MR/{digits(5)}/{ri(24, 26)}"])


def lab_id():
    return ch([f"LB{ri(2601, 2611)}-{digits(5)}", f"SMP/{ri(25, 26)}/{ri(1, 12):02d}/{digits(4)}",
               f"{ri(2601, 2611)}{digits(4)}", f"S{digits(8)}", f"BC-{digits(6)}"])


def nikshay_id():
    return ch([digits(8, "123"), digits(7, "123")])


def tb_no():
    return f"{ri(2025, 2026)}/{ch(['TBU', 'TU'])}-{ri(1, 9)}/{digits(4)}"


def rch_id():
    return not_verhoeff(12, "1" * 6 + "2345")


def pmjay_id():
    L = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    return ch([("P" + "".join(ch(L + DIG) for _ in range(8))),
               "".join(ch(L + DIG) for _ in range(9))]).upper()


def preauth_no():
    return f"PA/{ri(2601, 2611)}/{ch(['UP', 'BR', 'MP', 'RJ', 'JH', 'CG', 'HR', 'MH', 'GJ', 'KA', 'TN'])}/{digits(7)}"


def esj_id():
    return ch([f"ESJ-{ri(2601, 2611)}-{digits(7)}", f"HWC/{digits(4)}/{digits(6)}", f"TC{digits(10, '12345')}"])


def neg_number():
    return ch([f"Batch No. {ri(2300000, 2612999)}", f"Lot No. B{ri(23, 26)}K{ri(1000, 9999)}",
               f"Invoice No. {ch('12345')}{digits(9)}", f"Bill No. {ri(10, 26)}/{ri(1000, 9999)}",
               f"Receipt No. {ri(100000, 999999)}", f"Token No. {ri(1, 180)}",
               f"Ref. No. {ri(10000, 99999)}/{ri(24, 26)}", f"Vial No. {ri(1000, 9999)}",
               f"Sample tube barcode lot {ri(100000, 999999)}", f"Kit Lot {ch('ABCDEFG')}{ri(1000000, 9999999)}"])


# --------------------------------------------------------------------------
# Name pools (Latin script)
# --------------------------------------------------------------------------
POOLS = {
    "north": dict(
        m=["Ramesh", "Suresh", "Rajesh", "Mukesh", "Anil", "Sunil", "Vijay", "Sanjay", "Deepak", "Pankaj",
           "Ravi", "Amit", "Rohit", "Manoj", "Dinesh", "Ashok", "Rakesh", "Santosh", "Brijesh", "Jitendra",
           "Ramkishan", "Harish", "Shivam", "Ankit", "Gaurav", "Lalit", "Kailash", "Om Prakash",
           "Ram Naresh", "Shyam Sundar", "Prem", "Chhotelal", "Bablu", "Hari Shankar", "Mahesh", "Arvind"],
        f=["Sunita", "Geeta", "Anita", "Rekha", "Pooja", "Priya", "Kiran", "Manju", "Savitri", "Kamla",
           "Shanti", "Neelam", "Seema", "Rani", "Pushpa", "Meena", "Kusum", "Saroj", "Rinki", "Khushboo",
           "Anjali", "Nisha", "Preeti", "Babita", "Radha", "Leelawati", "Phoolmati", "Urmila"],
        sur=["Kumar", "Sharma", "Verma", "Yadav", "Gupta", "Singh", "Mishra", "Tiwari", "Pandey", "Chauhan",
             "Maurya", "Prajapati", "Patel", "Kushwaha", "Rajput", "Meena", "Jatav", "Paswan", "Mandal",
             "Sahu", "Srivastava", "Dubey", "Chaudhary", "Rawat", "Bisht", "Negi", "Nishad", "Rajbhar"],
        fsur=["Devi", "Devi", "Kumari"]),
    "muslim": dict(
        m=["Mohd. Iqbal|Iqbal", "Imran", "Salim", "Arif", "Shahid", "Irfan", "Rizwan", "Faizan", "Aslam",
           "Zubair", "Nadeem", "Javed", "Abdul Rahman|Abdul", "Mohammed Basheer|Basheer", "Anwar",
           "Md. Sajid|Sajid", "Naseem", "Shakeel"],
        f=["Shabana", "Rukhsana", "Nazia", "Farzana", "Ayesha", "Shaheen", "Reshma", "Zainab", "Nasreen",
           "Sabiha", "Tabassum", "Fatima", "Gulnaz", "Shagufta"],
        sur=["Ansari", "Khan", "Qureshi", "Siddiqui", "Shaikh", "Pathan", "Malik", "Mirza", "Alvi", "Rizvi",
             "Mansuri"],
        fsur=["Khatoon", "Begum", "Bano", "Parveen"]),
    "sikh": dict(
        m=["Gurpreet", "Harjeet", "Manpreet", "Jaswinder", "Kulwant", "Balwinder", "Sukhdev", "Harbhajan",
           "Amarjit", "Rajinder", "Gurnam", "Paramjit"],
        f=["Harpreet", "Manjeet", "Jasleen", "Simran", "Gurmeet", "Kulwinder", "Navneet", "Rajwant"],
        msur=["Singh", "Singh", "Singh Sandhu", "Singh Gill", "Singh Dhillon", "Singh Brar", "Singh Sidhu"],
        fsur=["Kaur", "Kaur", "Kaur Sandhu", "Kaur Gill", "Kaur Dhillon", "Kaur Brar"]),
    "west": dict(
        m=["Sachin", "Ganesh", "Sandeep", "Nitin", "Prashant", "Vishal", "Hitesh", "Jignesh", "Bhavesh",
           "Dattatray", "Mahesh", "Tukaram", "Kalpesh", "Rameshbhai|Ramesh", "Bharat"],
        f=["Swati", "Snehal", "Ashwini", "Manisha", "Hetal", "Komal", "Jyoti", "Vaishali", "Rupali", "Bhavna",
           "Sangita", "Manjulaben|Manjula", "Lata"],
        sur=["Patil", "Jadhav", "Pawar", "Shinde", "Deshmukh", "Kulkarni", "Joshi", "Patel", "Shah", "Desai",
             "Parmar", "Solanki", "More", "Gaikwad", "Bhil", "Damor", "Vasava"]),
    "ka": dict(
        m=["Manjunath", "Raghavendra", "Basavaraj", "Nagaraj", "Shivakumar", "Mallikarjun", "Prakash",
           "Naveen", "Siddappa", "Hanumanthappa"],
        f=["Lakshmi", "Savitha", "Shilpa", "Bhavana", "Roopa", "Kavya", "Gowramma", "Shobha", "Yellamma",
           "Renuka"],
        sur=["Gowda", "Shetty", "Hegde", "Patil", "Naik", "Reddy", "Kamath", "Rao", "Nayak", "Kulkarni"]),
    "ta": dict(
        m=["Murugan", "Senthil Kumar|Senthil", "Karthik", "Rajendran", "Sivakumar", "Balaji", "Arumugam",
           "Selvam", "Muthu", "Velmurugan", "Ramasamy"],
        f=["Selvi", "Meenakshi", "Revathi", "Saraswathi", "Sumathi", "Kalaiselvi", "Malathi", "Valli",
           "Pandiammal", "Jayanthi"],
        sur=["Krishnan", "Subramaniam", "Ramasamy", "Pandian", "Velu", "Perumal", "Chinnasamy"]),
    "te": dict(
        m=["Venkatesh", "Srinivas", "Ramesh Babu|Ramesh", "Narasimha", "Venkata Rao|Venkata", "Chandrasekhar",
           "Raju", "Satyanarayana", "Mallesh", "Yadagiri"],
        f=["Venkata Lakshmi|Lakshmi", "Padma", "Vani", "Sujatha", "Lalitha", "Anitha", "Swarupa", "Bhagya",
           "Sammakka", "Durga"],
        sur=["Reddy", "Naidu", "Rao", "Chowdary", "Goud", "Raju", "Varma", "Yadav", "Mudiraj"]),
    "kl": dict(
        m=["Suresh", "Rajeev", "Anil Kumar|Anil", "Sreejith", "Vineeth", "Unnikrishnan", "Biju", "Gopakumar",
           "Rajan", "Sasidharan"],
        f=["Deepa", "Anjali", "Sreelekha", "Lakshmi", "Remya", "Sindhu", "Bindu", "Ambika", "Sarojini"],
        sur=["Nair", "Menon", "Pillai", "Kurup", "Nambiar", "Panicker", "Warrier", "K.", "Das"]),
    "christian": dict(
        m=["Joseph", "Thomas", "George", "Mathew", "John", "Anthony", "Sebastian", "Paul", "Francis", "Jose",
           "Babu"],
        f=["Mary", "Annamma", "Rosamma", "Elizabeth", "Susan", "Jessy", "Sheeba", "Celine", "Grace", "Leena",
           "Molly"],
        sur=["Varghese", "Kurian", "Chacko", "Mathew", "Thomas", "Joseph", "Fernandes", "D'Souza", "Pinto",
             "Lobo", "Toppo", "Ekka", "Minj"]),
    "east": dict(
        m=["Subhas", "Bapi", "Sujit", "Prasenjit", "Debashis", "Manas Ranjan|Manas", "Biswajit", "Tapan",
           "Ashis", "Sanjib", "Pradeep", "Gour", "Bikram"],
        f=["Rina", "Mousumi", "Sasmita", "Tapasi", "Puja", "Mamata", "Lipika", "Sabita", "Jharna", "Swapna",
           "Bijayalaxmi"],
        sur=["Das", "Mondal", "Sarkar", "Ghosh", "Banerjee", "Mukherjee", "Pradhan", "Behera", "Sahoo",
             "Mohanty", "Nayak", "Roy", "Barik", "Mahato", "Hansda", "Murmu"]),
    "ne_as": dict(
        m=["Bikash", "Pranjal", "Dipankar", "Jayanta", "Hemanta", "Rupam", "Nayan", "Bhaskar"],
        f=["Bornali", "Jonali", "Dipika", "Pallabi", "Anamika", "Mridula", "Rashmi", "Junmoni"],
        sur=["Gogoi", "Bora", "Kalita", "Baruah", "Saikia", "Hazarika", "Deka", "Das", "Phukan"]),
    "ne_mz": dict(
        m=["Lalthanzuala", "Lalremruata", "Vanlalhruaia", "Zothanpuia", "Lalrinawma", "Lalmuanpuia"],
        f=["Lalrinpuii", "Lalhmangaihi", "Vanlalruati", "Zothansangi", "Malsawmtluangi", "Lalnunsiami"],
        sur=["Ralte", "Hmar", "Pachuau", "Sailo", "Chhangte", "Khiangte", "Renthlei"]),
    "ne_ml": dict(
        m=["Pynskhem", "Banteilang", "Wanshan", "Ricky", "Iohbor", "Kyrshan", "Saljrang"],
        f=["Daphisha", "Wanda", "Badalin", "Iarisha", "Bandarishisha|Banda", "Chimchi"],
        sur=["Lyngdoh", "Kharkongor", "Syiem", "Nongrum", "Marak", "Sangma", "Lyngwa", "Momin"]),
    "ne_mn": dict(
        m=["Ibomcha", "Tomba", "Ibochouba", "Premjit", "Rajen", "Bijoy"],
        f=["Ibemhal", "Thoibi", "Memcha", "Sanatombi", "Bembem", "Ongbi Leima|Leima"],
        sur=["Thangjam", "Laishram", "Ningthoujam", "Khumanthem", "Oinam", "Yumnam"]),
    "ne_ng": dict(
        m=["Kezhokhoto", "Vikhozo", "Thejavizo", "Akum", "Imtisunep", "Kevisetuo"],
        f=["Kenei", "Vitsono", "Aseno", "Akala", "Atsula", "Neikhrielie"],
        sur=["Sema", "Angami", "Kense", "Zeliang", "Sangtam", "Lotha", "Jamir"]),
    "ne_sk": dict(
        m=["Rinchen", "Tenzing", "Pemba", "Karma", "Sonam", "Nima"],
        f=["Pema", "Doma", "Yangchen", "Diki", "Choden", "Ongmu"],
        sur=["Bhutia", "Lepcha", "Sherpa", "Tamang", "Gurung", "Rai", "Subba"]),
}

# Devanagari pools
HI_POOLS = {
    "hi_hindu": dict(
        m=["रमेश", "सुरेश", "राजेश", "मुकेश", "अनिल", "विजय", "संजय", "दीपक", "रवि", "मनोज", "दिनेश", "अशोक",
           "राकेश", "संतोष", "बृजेश", "जितेंद्र", "रामकिशन", "शिवम", "अंकित", "ओमप्रकाश", "रामनरेश",
           "श्यामसुंदर", "लालजी", "छोटेलाल", "हरिशंकर", "बबलू", "महेश", "कैलाश"],
        f=["सुनीता", "गीता", "अनीता", "रेखा", "पूजा", "प्रिया", "किरन", "मंजू", "सावित्री", "कमला", "शांति",
           "नीलम", "सीमा", "रानी", "पुष्पा", "मीना", "कुसुम", "सरोज", "रिंकी", "खुशबू", "बबीता", "राधा",
           "लीलावती", "फूलमती", "उर्मिला"],
        sur=["कुमार", "शर्मा", "वर्मा", "यादव", "गुप्ता", "सिंह", "मिश्रा", "तिवारी", "पांडेय", "चौहान", "मौर्य",
             "प्रजापति", "पटेल", "कुशवाहा", "मीणा", "पासवान", "मंडल", "साहू", "श्रीवास्तव", "दुबे", "चौधरी",
             "रावत", "निषाद", "राजभर"],
        fsur=["देवी", "देवी", "कुमारी"]),
    "hi_muslim": dict(
        m=["मो. इरफ़ान|इरफ़ान", "सलीम", "आरिफ़", "इमरान", "शाहिद", "जावेद", "अनवर", "नसीम", "रिज़वान"],
        f=["शबाना", "रुख़साना", "नाज़िया", "फ़रज़ाना", "आयशा", "रेशमा", "नसरीन", "शाहीन"],
        sur=["अंसारी", "ख़ान", "कुरैशी", "सिद्दीकी", "मलिक", "मंसूरी"],
        fsur=["ख़ातून", "बेगम", "बानो", "परवीन"]),
    "hi_sikh": dict(
        m=["गुरप्रीत", "हरजीत", "मनप्रीत", "जसविंदर", "कुलवंत"],
        f=["हरप्रीत", "मनजीत", "जसलीन", "गुरमीत"],
        msur=["सिंह"], fsur=["कौर"]),
}

# rough Devanagari -> Latin transliteration (for ABHA address / e-mail handles)
_CONS = {"क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh",
         "ञ": "n", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d",
         "ध": "dh", "न": "n", "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r",
         "ल": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h", "ड़": "r", "ढ़": "rh", "क़": "q",
         "ख़": "kh", "ग़": "gh", "ज़": "z", "फ़": "f"}
_MATRA = {"ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ृ": "ri"}
_VOW = {"अ": "a", "आ": "a", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
        "ऋ": "ri"}


def translit(s):
    out = []
    chars = list(s)
    i = 0
    while i < len(chars):
        c = chars[i]
        if i + 1 < len(chars) and chars[i + 1] == "़":
            c = c + "़"
            i += 1
        nxt = chars[i + 1] if i + 1 < len(chars) else ""
        if c in _CONS:
            out.append(_CONS[c])
            if nxt in _MATRA:
                out.append(_MATRA[nxt])
                i += 1
            elif nxt == "्":
                i += 1
            elif nxt and (nxt in _CONS or (i + 2 < len(chars) and chars[i + 2] == "़")) :
                out.append("a")
            elif nxt in ("ं", "ँ"):
                out.append("a")
        elif c in _VOW:
            out.append(_VOW[c])
        elif c in ("ं", "ँ"):
            out.append("n")
        elif c == " ":
            out.append(" ")
        i += 1
    return "".join(out)


class Person:
    def __init__(self, given, last, g, pool, full, short, hindi=False, family=""):
        self.given, self.last, self.g, self.pool = given, last, g, pool
        self.full, self.short, self.hindi, self.family = full, short, hindi, family
        if hindi:
            self.rom_first = translit(short).replace(" ", "")
            self.rom_last = translit(last).replace(" ", "") if last else ""
        else:
            self.rom_first = re.sub(r"[^A-Za-z]", "", short)
            self.rom_last = re.sub(r"[^A-Za-z]", "", last.split()[-1]) if last else ""

    def initials(self):
        if not self.last or self.pool in ("ne_mn",):
            return self.full
        toks = [t for t in self.given.replace(".", " ").split() if t]
        sep = ch([".", ". ", " "])
        ini = sep.join(t[0] for t in toks) + ("." if sep != " " else "")
        return f"{ini.strip()} {self.last}"


def make_person(pool, g, family=None, hindi=False, avoid=()):
    PL = (HI_POOLS if hindi else POOLS)[pool]
    for _ in range(20):
        raw = ch(PL["m" if g == "M" else "f"])
        if "|" in raw:
            given, short = raw.split("|")
        else:
            given, short = raw, raw
        if given not in avoid:
            break
    if pool in ("sikh", "hi_sikh"):
        last = ch(PL["msur"] if g == "M" else PL["fsur"])
        fam = family or (last.split()[-1] if len(last.split()) > 1 else "")
        return Person(given, last, g, pool, f"{given} {last}", short, hindi, fam)
    if pool == "ne_mn":
        fam = family or ch(PL["sur"])
        last = "Singh" if g == "M" else "Devi"
        return Person(given, last, g, pool, f"{fam} {given} {last}", short, hindi, fam)
    if pool == "ta" and opt(0.6):
        ini = ch("ABGKMNPRSTV") + "."
        return Person(given, "", g, pool, f"{ini} {given}", short, hindi, ini)
    fam = family or ch(PL["sur"])
    last = fam
    if g == "F" and "fsur" in PL and opt(0.45):
        last = ch(PL["fsur"])
    if pool == "north" and g == "M" and last not in ("Kumar",) and opt(0.18) and " " not in given:
        given = given + " Kumar"
    if pool == "kl" and last == "K.":
        return Person(given, "K.", g, pool, f"{given} K.", short, hindi, fam)
    return Person(given, last, g, pool, f"{given} {last}", short, hindi, fam)


def relative(p, g):
    return make_person(p.pool, g, family=p.family or None, hindi=p.hindi, avoid=(p.given,))


def P(p, form="full", style=None):
    if form == "short":
        s = p.short
    elif form == "last":
        s = p.last if (p.last and p.last not in ("Devi", "Kumari", "K.") and p.pool != "ne_mn") else p.full
        if p.pool in ("sikh",):
            s = p.full
    elif form == "init":
        s = p.initials()
    else:
        s = p.full
    if style == "caps":
        s = s.upper()
    elif style == "lower":
        s = s.lower()
    return (s, "PERSON")


def nst():
    return ch([None] * 7 + ["caps", "caps", "lower"])


# --------------------------------------------------------------------------
# Places
# --------------------------------------------------------------------------
STATES = {
    "Uttar Pradesh": dict(vp="north", ab=["UP", "U.P.", "Uttar Pradesh", "Uttar Pradesh"], d={
        "Lucknow": (["Mohanlalganj", "Malihabad", "Bakshi Ka Talab"], "226", ["Aliganj", "Indira Nagar", "Chinhat", "Rajajipuram"]),
        "Ballia": (["Bansdih", "Rasra", "Bairia"], "277", None),
        "Gorakhpur": (["Campierganj", "Sahjanwa", "Chauri Chaura"], "273", None),
        "Barabanki": (["Haidergarh", "Fatehpur", "Ramnagar"], "225", None),
        "Bahraich": (["Nanpara", "Kaiserganj", "Mihinpurwa"], "271", None),
        "Varanasi": (["Pindra", "Cholapur", "Araziline"], "221", ["Lanka", "Shivpur", "Sigra"])}),
    "Bihar": dict(vp="north", ab=["Bihar"], d={
        "West Champaran": (["Bagaha", "Narkatiaganj", "Ramnagar"], "845", None),
        "Muzaffarpur": (["Kanti", "Motipur", "Sakra"], "843", None),
        "Gaya": (["Sherghati", "Tekari", "Bodh Gaya"], "824", None),
        "Darbhanga": (["Benipur", "Jale", "Bahadurpur"], "847", None),
        "Purnia": (["Banmankhi", "Dhamdaha", "Kasba"], "854", None),
        "Patna": (["Phulwari", "Danapur", "Masaurhi"], "800", ["Kankarbagh", "Rajendra Nagar", "Boring Road"])}),
    "Madhya Pradesh": dict(vp="north", ab=["MP", "M.P.", "Madhya Pradesh"], d={
        "Sehore": (["Ashta", "Ichhawar", "Budhni"], "466", None),
        "Rewa": (["Mauganj", "Sirmour", "Teonthar"], "486", None),
        "Jhabua": (["Petlawad", "Thandla", "Meghnagar"], "457", None),
        "Bhopal": (["Berasia", "Huzur"], "462", ["Kolar Road", "Bairagarh", "Ashoka Garden"])}),
    "Rajasthan": dict(vp="north", ab=["Rajasthan", "Raj."], d={
        "Barmer": (["Chohtan", "Balotra", "Sheo"], "344", None),
        "Alwar": (["Tijara", "Rajgarh", "Kishangarh Bas"], "301", None),
        "Bhilwara": (["Asind", "Mandal", "Shahpura"], "311", None),
        "Jaipur": (["Sanganer", "Chomu"], "302", ["Sanganer", "Malviya Nagar", "Jhotwara"])}),
    "Haryana": dict(vp="north", ab=["Haryana"], d={
        "Nuh": (["Punhana", "Ferozepur Jhirka", "Tauru"], "122", None),
        "Hisar": (["Hansi", "Barwala", "Adampur"], "125", None)}),
    "Jharkhand": dict(vp="north", ab=["Jharkhand"], d={
        "Ranchi": (["Bundu", "Kanke", "Ormanjhi"], "835", None),
        "Dumka": (["Jama", "Shikaripara", "Masalia"], "814", None)}),
    "Uttarakhand": dict(vp="north", ab=["Uttarakhand"], d={
        "Chamoli": (["Gairsain", "Karnaprayag", "Tharali"], "246", None),
        "Almora": (["Ranikhet", "Dwarahat", "Bhikiyasain"], "263", None)}),
    "Chhattisgarh": dict(vp="north", ab=["Chhattisgarh", "C.G."], d={
        "Bastar": (["Bakawand", "Tokapal", "Bastanar"], "494", None),
        "Raigarh": (["Gharghoda", "Kharsia", "Dharamjaigarh"], "496", None)}),
    "Delhi": dict(vp="north", ab=["New Delhi", "Delhi"], d={
        "South Delhi": ([], "110", ["Sangam Vihar", "Tigri Extension", "Dakshinpuri"]),
        "North East Delhi": ([], "110", ["Seelampur", "Jafrabad", "Bhajanpura"]),
        "West Delhi": ([], "110", ["Uttam Nagar", "Vikaspuri", "Najafgarh"])}),
    "Punjab": dict(vp="punjab", ab=["Punjab"], d={
        "Moga": (["Nihal Singh Wala", "Baghapurana", "Dharamkot"], "142", None),
        "Bathinda": (["Talwandi Sabo", "Rampura Phul", "Maur"], "151", None),
        "Gurdaspur": (["Batala", "Dera Baba Nanak", "Kalanaur"], "143", None)}),
    "Maharashtra": dict(vp="west", ab=["Maharashtra"], d={
        "Nashik": (["Igatpuri", "Sinnar", "Peth"], "422", None),
        "Yavatmal": (["Pusad", "Wani", "Ner"], "445", None),
        "Pune": (["Haveli", "Mulshi"], "411", ["Hadapsar", "Kothrud", "Yerawada"])}),
    "Gujarat": dict(vp="west", ab=["Gujarat"], d={
        "Dahod": (["Limkheda", "Jhalod", "Devgadh Baria"], "389", None),
        "Banaskantha": (["Tharad", "Deesa", "Vav"], "385", None),
        "Ahmedabad": (["Daskroi", "Sanand"], "380", ["Vatva", "Juhapura", "Naroda"])}),
    "Karnataka": dict(vp="ka", ab=["Karnataka"], d={
        "Raichur": (["Sindhanur", "Manvi", "Lingasugur"], "584", None),
        "Mandya": (["Maddur", "Malavalli", "Pandavapura"], "571", None),
        "Bengaluru Urban": (["Anekal", "Yelahanka"], "560", ["Jayanagar", "Yelahanka New Town", "Rajajinagar"])}),
    "Tamil Nadu": dict(vp="ta", ab=["Tamil Nadu", "TN"], d={
        "Villupuram": (["Tindivanam", "Gingee", "Vanur"], "604", None),
        "Madurai": (["Melur", "Usilampatti", "Peraiyur"], "625", None),
        "Dharmapuri": (["Pennagaram", "Harur", "Palacode"], "636", None),
        "Chennai": ([], "600", ["Tondiarpet", "Vyasarpadi", "Saidapet"])}),
    "Kerala": dict(vp="kl", ab=["Kerala"], d={
        "Malappuram": (["Tirur", "Perinthalmanna", "Ponnani"], "676", None),
        "Kottayam": (["Pala", "Vaikom", "Changanassery"], "686", None),
        "Idukki": (["Thodupuzha", "Devikulam", "Peerumade"], "685", None)}),
    "Andhra Pradesh": dict(vp="te", ab=["Andhra Pradesh", "A.P."], d={
        "Guntur": (["Tenali", "Mangalagiri", "Ponnur"], "522", None)}),
    "Telangana": dict(vp="te", ab=["Telangana", "TS"], d={
        "Nalgonda": (["Miryalaguda", "Devarakonda", "Nakrekal"], "508", None),
        "Adilabad": (["Utnoor", "Indervelli", "Boath"], "504", None)}),
    "West Bengal": dict(vp="east", ab=["West Bengal", "W.B."], d={
        "Murshidabad": (["Domkal", "Jangipur", "Kandi"], "742", None),
        "Purulia": (["Jhalda", "Manbazar", "Raghunathpur"], "723", None),
        "South 24 Parganas": (["Basanti", "Canning", "Kultali"], "743", None),
        "Kolkata": ([], "700", ["Tiljala", "Behala", "Kasba"])}),
    "Odisha": dict(vp="east", ab=["Odisha"], d={
        "Kalahandi": (["Dharamgarh", "Junagarh", "Kesinga"], "766", None),
        "Mayurbhanj": (["Baripada", "Rairangpur", "Karanjia"], "757", None),
        "Ganjam": (["Chhatrapur", "Aska", "Hinjilicut"], "761", None)}),
    "Assam": dict(vp="ne_as", ab=["Assam"], d={
        "Dibrugarh": (["Tingkhong", "Naharkatia", "Chabua"], "786", None),
        "Barpeta": (["Sarthebari", "Bajali", "Kalgachia"], "781", None),
        "Kamrup Metro": ([], "781", ["Beltola", "Chandmari", "Zoo Road"])}),
    "Mizoram": dict(vp="ne_mz", ab=["Mizoram"], d={
        "Aizawl": ([], "796", ["Chaltlang", "Zarkawt", "Bawngkawn"]),
        "Lunglei": ([], "796", ["Chanmari", "Serkawn"])}),
    "Meghalaya": dict(vp="ne_ml", ab=["Meghalaya"], d={
        "East Khasi Hills": (["Mawsynram", "Mylliem", "Mawphlang"], "793", ["Laitumkhrah", "Mawlai", "Nongthymmai"]),
        "West Garo Hills": (["Rongram", "Dadenggre"], "794", None)}),
    "Manipur": dict(vp="ne_mn", ab=["Manipur"], d={
        "Imphal West": ([], "795", ["Uripok", "Singjamei", "Keishampat"]),
        "Thoubal": (["Lilong", "Wangjing"], "795", None)}),
    "Nagaland": dict(vp="ne_ng", ab=["Nagaland"], d={
        "Kohima": ([], "797", ["Kohima Village", "Midland", "Forest Colony"])}),
    "Sikkim": dict(vp="ne_sk", ab=["Sikkim", "East Sikkim"], d={
        "Gangtok": ([], "737", ["Development Area", "Tadong", "Ranipool"])}),
}

VILL = {
    "north": ["Rampur Khurd", "Bhawanipur", "Kheri", "Devipur", "Naya Gaon", "Madhopur", "Lakhanpur", "Sikandarpur",
              "Barhpur", "Chakia", "Dhanaura", "Belwa", "Pipra", "Semra", "Kesarpur", "Bariyarpur", "Harraiya",
              "Sarai Mir", "Gopalpur Kalan", "Mahuli", "Ratanpura", "Bhagwanpur"],
    "punjab": ["Kot Ise Khan", "Dhudike", "Ghall Kalan", "Bilaspur", "Rode", "Takhtupura", "Chuhar Chak", "Lopon"],
    "gj": ["Limdi", "Navagam", "Moti Bandibar", "Kharedi", "Rampura", "Bhatwada", "Zalod Faliya", "Sanjeli"],
    "west": ["Wadgaon", "Pimpalgaon", "Sakegaon", "Undirwadi", "Khandala", "Bhatwadi", "Limdi", "Navagam",
             "Moti Bandibar", "Kharedi"],
    "ka": ["Hosahalli", "Kodihalli", "Mallapura", "Kyathanahalli", "Bannur", "Gowdanapalya", "Hirebidari"],
    "ta": ["Kottur", "Pudupalayam", "Melapalayam", "Keezhavalavu", "Karadipatti", "Ammapettai", "Valathi"],
    "te": ["Thimmapuram", "Peddapalli", "Chintalapudi", "Kondapur", "Ramannapeta", "Gudur", "Narketpally"],
    "kl": ["Edappal", "Kuttippuram", "Ramapuram", "Erattupetta", "Kanjirappally", "Vazhakulam", "Karimannoor"],
    "east": ["Kalinagar", "Gopalpur", "Haripur", "Natungram", "Balarampur", "Bhabanipur", "Sundarpur",
             "Jagannathpur", "Kusumpur"],
    "ne_as": ["Borbari", "Kakopathar", "Tingrai", "Bihpuria", "Kalgachia Pathar", "Nagaghuli"],
    "ne_mz": ["Chaltlang", "Zarkawt", "Bawngkawn"],
    "ne_ml": ["Mawkyrwat", "Mawlai Mawroh", "Nongpoh", "Rongram"],
    "ne_mn": ["Lilong Haoreibi", "Wangjing Lamding", "Khongjom"],
    "ne_ng": ["Kohima Village", "Jotsoma", "Kigwema"],
    "ne_sk": ["Tadong", "Rumtek", "Ranka"],
}
KL_HOUSE = ["Puthenpurayil", "Kizhakkethil", "Valiyaveettil", "Thekkedath", "Kunnumpurath", "Parambil",
            "Chirayil", "Mannil"]
LANDMARK = ["near Primary School", "behind Shiv Mandir", "opp. Panchayat Bhawan", "near Jama Masjid",
            "near water tank", "near Durga Mandir", "behind Gramin Bank", "near St. Mary's Church", "near bus stand",
            "opposite ration shop", "near Anganwadi"]

STATE_POOLS = {
    "north": [("north", 75), ("muslim", 20), ("sikh", 5)],
    "punjab": [("sikh", 85), ("north", 15)],
    "west": [("west", 80), ("muslim", 15), ("christian", 5)],
    "ka": [("ka", 80), ("muslim", 12), ("christian", 8)],
    "ta": [("ta", 80), ("muslim", 10), ("christian", 10)],
    "te": [("te", 85), ("muslim", 15)],
    "kl": [("kl", 45), ("christian", 30), ("muslim", 25)],
    "east": [("east", 75), ("muslim", 25)],
}


def wchoice(pairs):
    tot = sum(w for _, w in pairs)
    x = R.uniform(0, tot)
    for k, w in pairs:
        x -= w
        if x <= 0:
            return k
    return pairs[-1][0]


def pool_for(vp, hindi=False, state=None):
    if hindi:
        if state == "हरियाणा":
            return wchoice([("hi_hindu", 55), ("hi_muslim", 35), ("hi_sikh", 10)])
        return wchoice([("hi_hindu", 78), ("hi_muslim", 18), ("hi_sikh", 4)])
    if vp.startswith("ne_"):
        if vp == "ne_as":
            return wchoice([("ne_as", 80), ("muslim", 20)])
        return vp
    return wchoice(STATE_POOLS[vp])


# Hindi places
HI_STATES = {
    "उत्तर प्रदेश": dict(ab=["उत्तर प्रदेश", "उ.प्र.", "उत्तर प्रदेश"], d={
        "बलिया": (["बांसडीह", "रसड़ा", "बैरिया"], "277", None),
        "लखनऊ": (["मोहनलालगंज", "मलिहाबाद"], "226", ["अलीगंज", "इंदिरा नगर", "चिनहट"]),
        "गोरखपुर": (["कैम्पियरगंज", "सहजनवा"], "273", None),
        "बहराइच": (["नानपारा", "कैसरगंज"], "271", None),
        "सीतापुर": (["बिसवां", "महमूदाबाद"], "261", None)}),
    "बिहार": dict(ab=["बिहार"], d={
        "पश्चिम चंपारण": (["बगहा", "नरकटियागंज"], "845", None),
        "मुज़फ़्फ़रपुर": (["कांटी", "मोतीपुर"], "843", None),
        "गया": (["शेरघाटी", "टिकारी"], "824", None),
        "दरभंगा": (["बेनीपुर", "जाले"], "847", None),
        "पटना": (["फुलवारी", "मसौढ़ी"], "800", ["कंकड़बाग", "राजेंद्र नगर"])}),
    "मध्य प्रदेश": dict(ab=["मध्य प्रदेश", "म.प्र."], d={
        "सीहोर": (["आष्टा", "इछावर"], "466", None),
        "रीवा": (["मऊगंज", "सिरमौर"], "486", None),
        "झाबुआ": (["पेटलावद", "थांदला"], "457", None),
        "भोपाल": (["बैरसिया"], "462", ["कोलार रोड", "अशोका गार्डन"])}),
    "राजस्थान": dict(ab=["राजस्थान"], d={
        "बाड़मेर": (["चौहटन", "बालोतरा"], "344", None),
        "अलवर": (["तिजारा", "राजगढ़"], "301", None),
        "भीलवाड़ा": (["आसींद", "मांडल"], "311", None),
        "जयपुर": (["चौमूं"], "302", ["सांगानेर", "झोटवाड़ा"])}),
    "हरियाणा": dict(ab=["हरियाणा"], d={
        "नूंह": (["पुन्हाना", "फ़िरोज़पुर झिरका"], "122", None),
        "हिसार": (["हांसी", "बरवाला"], "125", None)}),
    "झारखंड": dict(ab=["झारखंड"], d={
        "रांची": (["बुंडू", "कांके"], "835", None),
        "दुमका": (["जामा", "शिकारीपाड़ा"], "814", None)}),
    "छत्तीसगढ़": dict(ab=["छत्तीसगढ़", "छ.ग."], d={
        "बस्तर": (["बकावंड", "तोकापाल"], "494", None),
        "रायगढ़": (["घरघोड़ा", "खरसिया"], "496", None)}),
    "उत्तराखंड": dict(ab=["उत्तराखंड"], d={
        "चमोली": (["गैरसैंण", "कर्णप्रयाग"], "246", None),
        "अल्मोड़ा": (["रानीखेत", "द्वाराहाट"], "263", None)}),
}
HI_VILL = ["रामपुर खुर्द", "भवानीपुर", "खेड़ी", "देवीपुर", "नयागांव", "माधोपुर", "लखनपुर", "सिकंदरपुर", "बरहपुर",
           "चकिया", "धनौरा", "बेलवा", "पिपरा", "सेमरा", "केसरपुर", "हरदिया", "महुली", "गोपालपुर कलां",
           "बरियारपुर", "सरायमीर", "रतनपुरा"]
HI_LANDMARK = ["प्राथमिक विद्यालय के पास", "शिव मंदिर के पीछे", "पंचायत भवन के सामने", "जामा मस्जिद के पास",
               "पानी की टंकी के पास", "आंगनबाड़ी केंद्र के पास"]

EN_STATE_W = ([("Uttar Pradesh", 10), ("Bihar", 8), ("Madhya Pradesh", 5), ("Rajasthan", 5), ("Haryana", 2),
               ("Jharkhand", 3), ("Uttarakhand", 2), ("Chhattisgarh", 2), ("Delhi", 4), ("Punjab", 4),
               ("Maharashtra", 6), ("Gujarat", 4), ("Karnataka", 5), ("Tamil Nadu", 6), ("Kerala", 5),
               ("Andhra Pradesh", 3), ("Telangana", 3), ("West Bengal", 6), ("Odisha", 4), ("Assam", 4),
               ("Mizoram", 2), ("Meghalaya", 2), ("Manipur", 2), ("Nagaland", 1), ("Sikkim", 1)])
MX_STATE_W = [("Uttar Pradesh", 10), ("Bihar", 8), ("Madhya Pradesh", 4), ("Rajasthan", 4), ("Delhi", 4),
              ("Haryana", 2), ("Jharkhand", 2), ("Uttarakhand", 1), ("Chhattisgarh", 1), ("Maharashtra", 2),
              ("Gujarat", 1), ("Punjab", 2)]


class Loc(dict):
    __getattr__ = dict.get


def make_loc(state, hindi=False):
    S = HI_STATES[state] if hindi else STATES[state]
    dname = ch(list(S["d"]))
    blocks, pin3, urb = S["d"][dname]
    urban = bool(urb) and (not blocks or opt(0.35))
    vp = "hi" if hindi else S["vp"]
    vpool = HI_VILL if hindi else VILL["gj" if state == "Gujarat" else vp]
    v = ch(urb) if urban else ch(vpool)
    po = ch([x for x in vpool if x != v])
    b = ch(blocks) if blocks else dname
    pin = pin3 + f"{ri(0, 3)}{ri(0, 9)}{ri(0, 9)}"
    return Loc(state=state, stv=ch(S["ab"]), district=dname, block=b, village=v, po=po, pin=pin,
               urban=urban, vp=vp, hindi=hindi)


def house():
    return ch([f"H.No. {ri(1, 250)}", f"House No. {ri(1, 99)}/{ri(1, 9)}", f"#{ri(1, 400)}", f"Plot No. {ri(1, 120)}",
               f"Qtr. No. {ch('ABCD')}-{ri(1, 80)}", f"H. No. {ri(1, 30)}-{ri(1, 9)}", f"Makan No. {ri(1, 90)}"])


def addr_en(L, pin_mode="in", short=False, hinglish=False):
    """Return (address_text, pin). pin_mode: 'in' (PIN inside span), 'none', 'sep' (caller emits PIN)."""
    V, PO, B, D, ST, vp = L.village, L.po, L.block, L.district, L.stv, L.vp
    w = ri(1, 15)
    if L.state == "Delhi":
        D = ch(["New Delhi", "Delhi", D])
    if hinglish and vp == "north" and not L.urban:
        core = ch([f"Gram {V}, Post {PO}, Thana {B}, Zila {D}", f"gaon {V}, block {B}, jila {D}",
                   f"Gram {V}, {ch(LANDMARK)}, Tehsil {B}, {D}", f"{V} gaon, post {PO}, {D} ({ST})"])
    elif vp == "north":
        if L.urban:
            core = ch([f"{house()}, Gali No. {ri(1, 12)}, {V}, {D}, {ST}", f"{house()}, Sector {ri(1, 30)}, {V}, {D}",
                       f"{ch('ABCDEFGHJK')}-{ri(1, 300)}/{ri(1, 9)}, {V}, {D}",
                       f"{house()}, {V}, {ch(LANDMARK)}, {D}"])
        elif short:
            core = ch([f"Vill. {V}, {D}", f"{V}, {B}, {D}", f"Vill {V} PO {PO} Dist {D}"])
        else:
            core = ch([f"Vill. {V}, Post {PO}, Tehsil {B}, Distt. {D}, {ST}",
                       f"Village {V}, P.O. {PO}, Block {B}, District {D}, {ST}",
                       f"{house()}, Ward No. {w}, {V}, Teh. {B}, Dist. {D}",
                       f"Vill & PO {V}, PS {B}, Dist {D}, {ST}",
                       f"{V}, {ch(LANDMARK)}, Block {B}, {D} ({ST})",
                       f"vill {V.lower()} po {PO.lower()} dist {D.lower()}",
                       f"Mohalla {ch(['Qazi Tola', 'Purani Bazar', 'Harijan Basti', 'Pathan Tola', 'Naya Tola'])}, Ward {w}, {V}, {D}, {ST}"])
    elif vp == "punjab":
        core = ch([f"Vill. {V}, P.O. {PO}, Tehsil {B}, Distt. {D}, Punjab", f"VPO {V}, Teh. {B}, Distt. {D}",
                   f"{house()}, Ward No. {w}, {V}, {D}, {ST}"])
    elif vp == "west":
        if L.urban:
            core = ch([f"Flat No. {ri(101, 804)}, {ch(['Sai Krupa Apartments', 'Shivneri Society', 'Om Sai Residency', 'Gokul Dham', 'Sukhshanti Nagar'])}, {V}, {D}",
                       f"{house()}, {V}, {D}, {ST}", f"Room No. {ri(1, 40)}, Chawl No. {ri(1, 12)}, {V}, {D}"])
        elif L.state == "Gujarat":
            core = ch([f"Mu. Po. {V}, Ta. {B}, Ji. {D}, Gujarat", f"At & Po {V}, Ta. {B}, Dist. {D}",
                       f"{ch(['Nishal', 'Mandir', 'Kuva', 'Talav'])} Faliyu, {V}, Ta. {B}, {D}"])
        else:
            core = ch([f"At Post {V}, Tal. {B}, Dist. {D}, {ST}", f"Mu. {V}, Post {PO}, Ta. {B}, Ji. {D}",
                       f"{house()}, {V}, Tal. {B}, {D}"])
    elif vp == "ka":
        if L.urban:
            core = f"No. {ri(1, 400)}, {ri(1, 12)}th Cross, {ri(1, 9)}th Main, {V}, Bengaluru"
        else:
            core = ch([f"{house()}, {V}, {B} Taluk, {D} District, Karnataka", f"{V} (V), {B} Tq, {D} Dist",
                       f"Door No. {ri(1, 200)}, {V} village, {B} taluk, {D}"])
    elif vp == "ta":
        street = ch(["Pillaiyar Koil Street", "Main Road", "North Street", "Gandhi Nagar 2nd Street",
                     "Mariamman Koil Street", "Kamarajar Salai"])
        if L.urban:
            core = f"No. {ri(1, 150)}, {street}, {V}, Chennai"
        else:
            core = ch([f"Door No. {ri(1, 9)}/{ri(10, 300)}, {street}, {V}, {B} Taluk, {D} District, Tamil Nadu",
                       f"{ri(1, 120)}, {street}, {V} village, {B} Tk, {D} Dt"])
    elif vp == "te":
        core = ch([f"H.No. {ri(1, 9)}-{ri(1, 120)}/{ri(1, 9)}, {V} (V), {B} (M), {D} (Dt), {ST}",
                   f"D.No. {ri(1, 30)}-{ri(1, 99)}, {V} village, {B} mandal, {D} district"])
    elif vp == "kl":
        hn = ch(KL_HOUSE)
        core = ch([f"{hn} House, {V} P.O., {B}, {D} District, Kerala", f"{hn}, {V}, {D} Dt.",
                   f"{hn} (H), {V}, {B} Taluk, {D}"])
    elif vp == "east":
        if L.urban:
            core = f"{ri(1, 99)}/{ri(1, 9)}, {ch(['Tiljala Road', 'Diamond Harbour Road', 'Ram Krishna Pally Lane', 'Picnic Garden Road'])}, {V}, Kolkata"
        elif L.state == "Odisha":
            core = ch([f"At/Po {V}, Block {B}, Dist {D}, Odisha", f"At {V}, Po {PO}, PS {B}, Dist {D}"])
        else:
            core = ch([f"Vill+PO {V}, PS {B}, Dist {D}, West Bengal", f"Vill. {V}, P.O. {PO}, P.S. {B}, Dist. {D}"])
    elif vp == "ne_as":
        if L.urban:
            core = f"House No. {ri(1, 80)}, {ch(['Bylane 3', 'Lachit Path', 'Rupnagar Path'])}, {V}, Guwahati, Assam"
        else:
            core = ch([f"Vill. {V}, P.O. {PO}, P.S. {B}, Dist. {D}, Assam", f"{V} gaon, {B}, {D}"])
    elif vp == "ne_mz":
        core = ch([f"{house()}, {V}, {D}, Mizoram", f"{V} Veng, {D}, Mizoram"])
    elif vp == "ne_ml":
        core = ch([f"{V}, Shillong, {D}, Meghalaya", f"Vill. {V}, {B} C&RD Block, {D}"])
    elif vp == "ne_mn":
        core = ch([f"{V} Mayai Leikai, {D}, Manipur", f"{V}, {B}, {D}, Manipur"])
    elif vp == "ne_ng":
        core = f"Ward No. {w}, {V}, Kohima, Nagaland"
    else:
        core = f"{V}, Gangtok, {ST}"
    pin = L.pin
    if pin_mode == "in":
        core = core + ch([f" - {pin}", f" {pin}", f", PIN {pin}", f", Pin-{pin}", f" – {pin}", f", PIN: {pin}",
                          f"-{pin}", f", Pin code {pin}"])
    return core, pin


def addr_hi(L, pin_mode="in", dev=False):
    V, PO, B, D, ST = L.village, L.po, L.block, L.district, L.stv
    n, w = str(ri(1, 150)), str(ri(1, 15))
    if L.urban:
        core = ch([f"मकान नं. {n}, गली नं. {ri(1, 12)}, {V}, {D}", f"{n}, {V}, {D} ({ST})",
                   f"मकान नं. {n}, {V}, {ch(HI_LANDMARK)}, {D}"])
    else:
        core = ch([f"ग्राम {V}, पोस्ट {PO}, तहसील {B}, जिला {D}, {ST}",
                   f"ग्राम व पोस्ट {V}, थाना {B}, जिला {D} ({ST})",
                   f"मकान नं. {n}, वार्ड नं. {w}, {V}, ब्लॉक {B}, जिला {D}",
                   f"ग्राम-{V}, पो.-{PO}, जि.-{D}",
                   f"{V}, {ch(HI_LANDMARK)}, {B}, {D}",
                   f"ग्राम {V}, डाकघर {PO}, प्रखंड {B}, जिला {D}, {ST}"])
    pin = L.pin
    if pin_mode == "in":
        core = core + ch([f" - {pin}", f", पिन {pin}", f" पिन कोड {pin}", f"-{pin}", f", पिन- {pin}"])
    if dev:
        core, pin = deva(core), deva(pin)
    return core, pin


# --------------------------------------------------------------------------
# Facilities (never labelled; several carry person-like names on purpose)
# --------------------------------------------------------------------------
NAMED_FAC = ["Ram Manohar Lohia Hospital, New Delhi", "Dr. Ram Manohar Lohia Institute of Medical Sciences, Lucknow",
             "Lok Nayak Jai Prakash Narayan Hospital, New Delhi", "Sanjay Gandhi Postgraduate Institute of Medical Sciences, Lucknow",
             "Rajiv Gandhi Government General Hospital, Chennai", "Indira Gandhi Institute of Medical Sciences, Patna",
             "Sardar Vallabhbhai Patel Hospital, Ahmedabad", "Netaji Subhas Chandra Bose Medical College, Jabalpur",
             "Pt. B.D. Sharma PGIMS, Rohtak", "Lala Lajpat Rai Memorial Medical College, Meerut",
             "Guru Teg Bahadur Hospital, Delhi", "Deen Dayal Upadhyay Hospital, Hari Nagar",
             "Maharaja Yeshwantrao Hospital, Indore", "Dr. Bhimrao Ambedkar Memorial Hospital, Raipur",
             "Anugrah Narayan Magadh Medical College & Hospital, Gaya", "Rajendra Institute of Medical Sciences, Ranchi",
             "Sri Krishna Medical College & Hospital, Muzaffarpur", "Jawaharlal Nehru Medical College, Belagavi",
             "Mahatma Gandhi Memorial Hospital, Warangal", "Sanjay Gandhi Memorial Hospital, Rewa",
             "Lokmanya Tilak Municipal General Hospital, Sion", "Babu Jagjivan Ram Memorial Hospital, Jahangirpuri",
             "St. John's Medical College Hospital, Bengaluru", "Kasturba Gandhi Hospital, Gangtok",
             "Sri Venkateswara Ramnarain Ruia Government General Hospital, Tirupati",
             "Acharya Vinoba Bhave Rural Hospital, Wardha", "Guru Gobind Singh Medical College, Faridkot",
             "Chacha Nehru Bal Chikitsalaya, Delhi"]


def fac_en(L, kind="any"):
    gen_big = [f"District Hospital, {L.district}", f"Government Medical College & Hospital, {L.district}",
               f"Civil Hospital, {L.district}", f"District Women Hospital, {L.district}",
               f"{ch(['Sai', 'Shri Balaji', 'Lifeline', 'Apex', 'Sanjeevani', 'Arogya', 'City', 'Sunrise'])} Hospital, {L.district}"]
    gen_small = [f"CHC {L.block}", f"Community Health Centre, {L.block}", f"PHC {L.po}", f"Primary Health Centre, {L.po}",
                 f"Sub-District Hospital, {L.block}", f"AB-HWC {L.po}", f"Urban PHC {L.district}"]
    if kind == "big":
        return ch(NAMED_FAC) if opt(0.35) else ch(gen_big)
    if kind == "small":
        return ch(gen_small)
    return ch(NAMED_FAC) if opt(0.2) else ch(gen_big + gen_small)


HI_NAMED_FAC = ["डॉ. राम मनोहर लोहिया आयुर्विज्ञान संस्थान, लखनऊ", "लाल बहादुर शास्त्री अस्पताल, दिल्ली",
                "दीन दयाल उपाध्याय जिला चिकित्सालय", "राजेंद्र आयुर्विज्ञान संस्थान, रांची",
                "महाराजा यशवंतराव अस्पताल, इंदौर", "संजय गांधी स्मारक चिकित्सालय, रीवा",
                "अनुग्रह नारायण मगध मेडिकल कॉलेज, गया", "सरदार पटेल मेडिकल कॉलेज, बीकानेर",
                "बाबा राघव दास मेडिकल कॉलेज, गोरखपुर", "पं. जवाहरलाल नेहरू स्मृति चिकित्सा महाविद्यालय, रायपुर"]


def fac_hi(L, kind="any"):
    big = [f"जिला अस्पताल, {L.district}", f"जिला महिला चिकित्सालय, {L.district}", f"राजकीय मेडिकल कॉलेज, {L.district}"]
    small = [f"सामुदायिक स्वास्थ्य केंद्र, {L.block}", f"सीएचसी {L.block}", f"प्राथमिक स्वास्थ्य केंद्र, {L.po}",
             f"पीएचसी {L.po}", f"उपकेंद्र {L.po}", f"आयुष्मान आरोग्य मंदिर, {L.po}"]
    if kind == "big":
        return ch(HI_NAMED_FAC) if opt(0.35) else ch(big)
    if kind == "small":
        return ch(small)
    return ch(HI_NAMED_FAC) if opt(0.2) else ch(big + small)


ADULT_WARDS = ["Medicine Ward", "Female Medical Ward", "Male Surgical Ward", "ICU", "HDU", "Ward 3B", "Isolation Ward"]
WARDS = ["Medicine Ward", "Female Medical Ward", "Male Surgical Ward", "ICU", "HDU", "Ward 3B", "Maternity Ward",
         "Ortho Ward", "Isolation Ward", "Paediatric Ward", "Labour Room"]


# --------------------------------------------------------------------------
# Case (shared context for a note)
# --------------------------------------------------------------------------
class Case:
    def __init__(self, lang="en", g=None, age=(18, 80), state=None, pool=None):
        self.lang = lang
        hindi = lang == "hi"
        if state is None:
            if hindi:
                state = ch(list(HI_STATES))
            else:
                state = wchoice(MX_STATE_W if lang == "mixed" else EN_STATE_W)
        self.state = state
        self.loc = make_loc(state, hindi)
        self.pool = pool or pool_for(self.loc.vp, hindi, state)
        self.g = g or ch("MF")
        self.pat = make_person(self.pool, self.g, hindi=hindi)
        self.age = ri(*age) if isinstance(age, tuple) else age
        self.ref = rand_date(2026)
        self.dob = self.ref - timedelta(days=self.age * 365 + ri(0, 360))
        self.hindi = hindi
        self._rel = None

    def staff(self, g=None, pool=None):
        g = g or ch("MF")
        if self.hindi:
            pool = pool or wchoice([("hi_hindu", 85), ("hi_muslim", 10), ("hi_sikh", 5)])
            return make_person(pool, g, hindi=True)
        if pool is None:
            pool = pool_for(self.loc.vp) if opt(0.6) else ch(["north", "muslim", "west", "ka", "ta", "te", "kl",
                                                               "east", "christian", "sikh", "ne_as"])
        return make_person(pool, g)

    def guardian(self):
        """(kind, person). kind in father/husband."""
        if self._rel:
            return self._rel
        if self.g == "F" and self.age >= 20 and opt(0.8):
            self._rel = ("husband", relative(self.pat, "M"))
        else:
            self._rel = ("father", relative(self.pat, "M"))
        return self._rel


def hon_en(p, married=True, prob=0.75):
    if not opt(prob):
        return ""
    if p.g == "M":
        opts = ["Mr. ", "Mr. ", "Mr ", "Shri ", "Sri "] + (["Thiru "] if p.pool == "ta" else [])
    elif married:
        opts = ["Mrs. ", "Smt. ", "Smt ", "Ms. "] + (["Tmt. "] if p.pool == "ta" else []) + \
               (["Mst. "] if p.pool == "muslim" else [])
    else:
        opts = ["Ms. ", "Miss ", "Km. "]
    return ch(opts)


def hon_hi(p, married=True, prob=0.7):
    if not opt(prob):
        return ""
    if p.g == "M":
        return ch(["श्री ", "श्री ", "श्री "])
    return ch(["श्रीमती ", "श्रीमती "]) if married else ch(["कु. ", "सुश्री "])


def dr_en():
    return ch(["Dr. ", "Dr. ", "Dr ", "DR. "])


def guardian_field_en(c):
    kind, rel = c.guardian()
    if kind == "husband":
        lab = ch(["W/o", "Husband's Name", "Husband", "Spouse", "W/O"])
    else:
        lab = ch(["S/o", "Father's Name", "F/N", "S/O"]) if c.g == "M" else ch(["D/o", "Father's Name", "D/O"])
    return (lab, [hon_en(rel, prob=0.3), P(rel)])


def guardian_inline_en(c):
    kind, rel = c.guardian()
    if kind == "husband":
        w = ch(["W/o", "wife of"])
    else:
        w = ch(["S/o", "son of"]) if c.g == "M" else ch(["D/o", "daughter of"])
    return [w, " ", hon_en(rel, prob=0.3), P(rel)]


def agesex_en(c):
    s = "M" if c.g == "M" else "F"
    sl = "Male" if c.g == "M" else "Female"
    return ch([f"{c.age}/{s}", f"{c.age} Y / {sl}", f"{c.age} yrs/{s}", f"{c.age}Y/{s}", f"{c.age} years, {sl}",
               f"{c.age}{s}"])


L_NAME = ["Name", "Name", "Patient Name", "Pt. Name", "Name of Patient", "Patient", "Pateint Name", "Patient's Name"]
L_ADDR = ["Address", "Address", "Addr.", "Adress", "Residential Address", "Permanent Address", "R/o", "Add"]
L_MOB = ["Mobile", "Mob.", "Mob No.", "Mobile No.", "Contact No.", "Phone", "Ph.", "Mobil no", "Contact"]
L_AADH = ["Aadhaar No.", "Aadhaar", "Aadhar No", "UID No.", "Aadhaar (UIDAI)", "Adhar no.", "AADHAAR"]
L_ABHA = ["ABHA No.", "ABHA Number", "ABHA ID", "Health ID (ABHA)", "ABHA"]
L_ABHAA = ["ABHA Address", "PHR Address", "ABHA addr", "ABHA address"]
L_DOB = ["DOB", "D.O.B.", "Date of Birth", "DoB", "Birth date"]


def aadhaar_field(dev=False):
    return (ch(L_AADH), [E(fmt_aadhaar(dev=dev), "AADHAAR")])


def abha_fields(c, both=None):
    if opt(0.15):
        return [(ch(L_ABHAA), [E(abha_address(c.pat, c.dob.year), "ABHA_ADDRESS")])]
    out = [(ch(L_ABHA), [E(abha_number(), "ABHA_NUMBER")])]
    if both or (both is None and opt(0.75)):
        out.append((ch(L_ABHAA), [E(abha_address(c.pat, c.dob.year), "ABHA_ADDRESS")]))
    return out


def idproof_field(p):
    k = ch(["aadhaar", "voter", "voter", "pan", "pan"])
    if k == "aadhaar":
        return aadhaar_field()
    if k == "voter":
        return (ch(["Voter ID", "EPIC No.", "ID proof (Voter card)", "Voter ID No."]), [E(epic_no(), "VOTER_ID")])
    return (ch(["PAN", "PAN No.", "ID proof (PAN)"]), [E(pan_no(p), "PAN")])


# --------------------------------------------------------------------------
# Clinical content (hard negatives: drugs, doses, labs, eponyms, dates)
# --------------------------------------------------------------------------
def hb():
    return f"{R.uniform(6.0, 13.5):.1f}"


def plt():
    return inr(ri(28, 420) * 1000)


def scen_en(g, preg=False):
    S = [
        dict(dx="Plasmodium vivax malaria", cc=f"fever with chills and rigors for {ri(3, 7)} days",
             exam=f"Temp {R.uniform(100, 103.5):.1f} F, PR {ri(90, 118)}/min, pallor +, spleen just palpable",
             labs=["MP smear: P. vivax ring forms seen", f"Hb {hb()} g/dL", f"Platelet {plt()} /cmm"],
             rx=["Tab. Chloroquine 250 mg - 4 tabs D1, 4 tabs D2, 2 tabs D3", "Tab. Primaquine 15 mg OD x 14 days",
                 "Tab. Paracetamol 650 mg SOS"], inv=["CBC", "MP-RDT", "G6PD"],
             course="Afebrile from day 3, tolerating orally."),
        dict(dx="Dengue fever with thrombocytopenia (NS1 positive)", cc=f"high grade fever, retro-orbital pain and myalgia x {ri(3, 5)} days",
             exam=f"BP {ri(96, 118)}/{ri(60, 78)} mmHg, petechiae over forearms, tourniquet test positive",
             labs=["NS1 Ag: positive", f"Platelet {inr(ri(18, 90) * 1000)} /cmm", f"HCT {ri(38, 48)} %", f"TLC {inr(ri(22, 48) * 100)} /cmm"],
             rx=["Tab. Paracetamol 500 mg QID (no NSAIDs)", "ORS 2-3 L/day", "Papaya leaf extract NOT advised"],
             inv=["CBC daily", "NS1/IgM dengue"], course="Platelet nadir on day 5, rose thereafter; no bleeding."),
        dict(dx="Type 2 diabetes mellitus with left diabetic foot ulcer", cc="non-healing ulcer left great toe x 3 weeks",
             exam="Wagner grade 2 ulcer, dorsalis pedis palpable", labs=[f"RBS {ri(220, 420)} mg/dL", f"HbA1c {R.uniform(8.1, 12.5):.1f} %",
                                                                          f"S. creatinine {R.uniform(0.8, 1.9):.1f} mg/dL"],
             rx=["Inj. Insulin Glargine 14 U SC HS", "Inj. Human Actrapid as per sliding scale", "Tab. Linezolid 600 mg BD x 10 days",
                 "Daily dressing with povidone iodine"], inv=["X-ray left foot", "pus C/S"],
             course="Wound debrided on day 2; slough reduced."),
        dict(dx="Acute gastroenteritis with moderate dehydration", cc=f"loose stools {ri(8, 14)} episodes and vomiting since last night",
             exam="Sunken eyes, skin pinch goes back slowly, PR 112/min", labs=[f"Na {ri(128, 136)} mEq/L", f"K {R.uniform(2.9, 3.8):.1f} mEq/L",
                                                                              f"Blood urea {ri(38, 78)} mg/dL"],
             rx=["ORS 200 ml after each loose stool", "Tab. Zinc 20 mg OD x 14 days", "Inj. Ondansetron 4 mg IV SOS",
                 "IV RL 1 L over 2 h"], inv=["Stool R/M", "S. electrolytes"], course="Rehydrated; Ryle's tube not required."),
        dict(dx="Pulmonary Koch's (new, microbiologically confirmed)", cc="cough with expectoration > 2 weeks, evening rise of temperature, weight loss",
             exam=f"Wt {ri(36, 52)} kg, coarse crepts right upper zone", labs=["CBNAAT: MTB detected, Rif resistance not detected",
                                                                              f"ESR {ri(40, 95)} mm/1st hr", "HIV: non-reactive"],
             rx=["4FDC (HRZE) 3 tabs OD empty stomach", "Tab. Pyridoxine 10 mg OD"], inv=["Sputum CBNAAT", "CXR PA view"],
             course="Started on ATT; tolerating well."),
        dict(dx="Community acquired pneumonia, right lower lobe", cc=f"fever, productive cough and breathlessness x {ri(4, 8)} days",
             exam=f"RR {ri(24, 32)}/min, SpO2 {ri(88, 94)}% on RA, bronchial breath sounds right infra-scapular",
             labs=[f"TLC {inr(ri(120, 220) * 100)} /cmm", f"CRP {ri(48, 160)} mg/L", "CXR: right lower zone consolidation"],
             rx=["Inj. Ceftriaxone 1 g IV BD", "Tab. Azithromycin 500 mg OD x 5 days", "Neb. Salbutamol 6 hourly"],
             inv=["CXR", "Sputum gram stain"], course="Oxygen weaned off on day 3."),
        dict(dx="Hypertensive urgency", cc="headache and giddiness, stopped antihypertensives 2 weeks back",
             exam=f"BP {ri(180, 210)}/{ri(110, 124)} mmHg, fundus grade II changes", labs=[f"S. creatinine {R.uniform(0.9, 1.6):.1f} mg/dL",
                                                                                         "ECG: LVH by voltage", "Urine albumin: trace"],
             rx=["Tab. Amlodipine 10 mg OD", "Tab. Telmisartan 40 mg OD", "Tab. Atorvastatin 10 mg HS"],
             inv=["Lipid profile", "ECG", "RFT"], course="BP settled to 150/90 mmHg over 24 h."),
        dict(dx="COPD with acute exacerbation", cc="increased breathlessness and wheeze, chulha smoke exposure",
             exam=f"SpO2 {ri(84, 90)}% on RA, bilateral wheeze", labs=["ABG: pH 7.34, pCO2 52 mmHg", f"Hb {hb()} g/dL"],
             rx=["Neb. Salbutamol + Ipratropium 6 hourly", "Inj. Hydrocortisone 100 mg IV TDS", "Tab. Prednisolone 40 mg OD x 5 days",
                 "Tiotropium rotacap 18 mcg OD"], inv=["CXR", "Spirometry after recovery"], course="Improved; on room air at discharge."),
        dict(dx="Neurotoxic snake bite", cc="bitten on right foot while working in the field at night, ptosis",
             exam="Bilateral ptosis, single fang mark right foot, no local swelling", labs=["20 min WBCT: clotted", f"Platelet {plt()} /cmm"],
             rx=["Inj. ASV 10 vials in NS over 1 h", "Inj. Neostigmine 0.5 mg + Atropine 0.6 mg", "Inj. Td 0.5 ml IM"],
             inv=["20 WBCT 6 hourly"], course="Ptosis resolved by 18 h; no ventilatory support needed."),
        dict(dx="Organophosphate poisoning (suicidal intent)", cc="consumed pesticide after family quarrel, vomiting, excess salivation",
             exam="Pupils pinpoint, fasciculations, GCS 13/15", labs=["S. pseudocholinesterase low", f"Na {ri(132, 140)} mEq/L"],
             rx=["Inj. Atropine titrated", "Inj. PAM 1 g IV 8 hourly", "Gastric lavage done"],
             inv=["RFT", "LFT"], course="Atropine tapered over 72 h; psychiatry referral done."),
        dict(dx="Enteric fever", cc=f"step-ladder fever x {ri(7, 12)} days, abdominal discomfort",
             exam="Coated tongue, mild hepatomegaly, relative bradycardia", labs=["Widal: TO 1:320, TH 1:160", "Blood C/S: S. Typhi",
                                                                                f"SGPT {ri(55, 140)} IU/L"],
             rx=["Inj. Ceftriaxone 2 g IV OD x 10 days", "Tab. Azithromycin 500 mg BD"], inv=["Blood culture", "LFT"],
             course="Defervescence by day 5."),
        dict(dx="Right Bell's palsy", cc="deviation of mouth to left and inability to close right eye since morning",
             exam="LMN type right facial palsy, no other deficit", labs=[f"RBS {ri(90, 140)} mg/dL"],
             rx=["Tab. Prednisolone 60 mg OD x 5 days then taper", "Tab. Valacyclovir 1 g TDS x 7 days", "Eye patch and lubricant drops"],
             inv=["RBS"], course="Partial improvement; physiotherapy taught."),
        dict(dx="Fracture neck of femur (left) - Austin Moore hemiarthroplasty done", cc="fall at home, unable to bear weight",
             exam="Left lower limb shortened and externally rotated", labs=[f"Hb {hb()} g/dL", "X-ray: displaced fracture NOF left"],
             rx=["Tab. Aceclofenac + Paracetamol BD x 5 days", "Inj. Enoxaparin 40 mg SC OD x 10 days", "Tab. Calcium + Vit D3 OD"],
             inv=["X-ray pelvis"], course="Mobilised with walker on POD 2; Foley's catheter removed."),
        dict(dx="Scrub typhus", cc="fever with headache x 8 days, eschar noted in axilla",
             exam="Eschar left axilla, generalised lymphadenopathy", labs=["Scrub typhus IgM ELISA: positive", f"SGOT {ri(60, 180)} IU/L",
                                                                           f"Platelet {plt()} /cmm"],
             rx=["Tab. Doxycycline 100 mg BD x 7 days", "Tab. Paracetamol 650 mg SOS"], inv=["Scrub IgM"],
             course="Afebrile within 48 h of doxycycline."),
        dict(dx="Acute exacerbation of bronchial asthma", cc="wheeze and cough since 2 days, Murphy's sign negative, no fever",
             exam="Bilateral rhonchi, PEFR 180 L/min", labs=[f"Eosinophils {ri(6, 14)} %", f"SpO2 {ri(90, 95)}% on RA"],
             rx=["Formoterol + Budesonide 6/200 rotacap BD", "Tab. Montelukast 10 mg HS", "Tab. Prednisolone 30 mg OD x 5 days"],
             inv=["PEFR"], course="Settled with nebulisation."),
    ]
    if g == "F":
        S += [
            dict(dx="Severe anaemia in pregnancy (G2P1, 32 weeks)", cc="breathlessness on exertion and easy fatigability",
                 exam=f"Pallor +++, pedal oedema, fundal height {ri(28, 32)} cm", labs=[f"Hb {R.uniform(5.2, 6.9):.1f} g/dL",
                                                                                       "PS: microcytic hypochromic", f"FHR {ri(132, 150)} bpm"],
                 rx=["Inj. Iron sucrose 200 mg in 100 ml NS (3 doses)", "Tab. IFA 1 BD", "Tab. Calcium 500 mg BD"],
                 inv=["CBC", "Hb electrophoresis"], course="1 unit PRBC transfused; Hb rose to 8.1 g/dL."),
            dict(dx="Pre-eclampsia with severe features, G2P1L1 at 36 weeks", cc="headache and swelling of feet x 4 days",
                 exam=f"BP {ri(160, 176)}/{ri(110, 118)} mmHg, urine albumin ++", labs=[f"Platelet {plt()} /cmm", f"SGPT {ri(30, 90)} IU/L",
                                                                                         "Urine albumin: 2+"],
                 rx=["Inj. Magnesium sulphate (Pritchard regimen)", "Tab. Labetalol 200 mg TDS", "Tab. Nifedipine 10 mg SOS"],
                 inv=["PIH profile", "NST"], course="Delivered by LSCS; baby to SNCU for observation."),
        ]
    return ch(S)


def lab_panel():
    rows = [("Haemoglobin", hb(), "g/dL", "12.0 - 15.0"), ("Total Leucocyte Count", inr(ri(38, 190) * 100), "/cmm", "4,000 - 11,000"),
            ("Platelet Count", plt(), "/cmm", "1,50,000 - 4,50,000"), ("ESR", str(ri(8, 90)), "mm/1st hr", "0 - 20"),
            ("Random Blood Sugar", str(ri(78, 380)), "mg/dL", "70 - 140"), ("Serum Creatinine", f"{R.uniform(0.6, 3.1):.2f}", "mg/dL", "0.6 - 1.2"),
            ("Blood Urea", str(ri(14, 96)), "mg/dL", "15 - 40"), ("SGPT (ALT)", str(ri(12, 160)), "IU/L", "< 40"),
            ("TSH", f"{R.uniform(0.3, 9.8):.2f}", "µIU/mL", "0.4 - 4.5"), ("HbA1c", f"{R.uniform(5.1, 12.4):.1f}", "%", "< 5.7"),
            ("Serum Sodium", str(ri(124, 146)), "mEq/L", "135 - 145"), ("Serum Potassium", f"{R.uniform(2.8, 5.9):.1f}", "mEq/L", "3.5 - 5.1"),
            ("HBsAg", ch(["Non-reactive", "Reactive"]), "", ""), ("Widal (TO)", ch(["1:80", "1:160", "1:320"]), "", "< 1:80")]
    R.shuffle(rows)
    return rows[:ri(4, 8)]


# --------------------------------------------------------------------------
# ENGLISH TEMPLATES
# --------------------------------------------------------------------------
def att_phrase_en(c, who=None):
    """Narrative mention of an attendant without honorific."""
    p = c.pat
    rel_kind = ch(["son", "daughter", "brother", "husband" if c.g == "F" else "wife", "nephew", "daughter-in-law",
                   "neighbour", "sister"])
    g = "F" if rel_kind in ("daughter", "wife", "daughter-in-law", "sister") else "M"
    if rel_kind == "neighbour":
        g = ch("MF")
        a = c.staff(g, pool=c.pool)
    else:
        a = relative(p, g)
    his = "his" if c.g == "M" else "her"
    form = ch(["full", "short", "short"])
    pre = ch([f"brought by {his} {rel_kind} ", f"accompanied by {rel_kind} ", f"{his.capitalize()} {rel_kind} ",
              f"attendant: {rel_kind} "])
    return pre, a, form, rel_kind


def en_discharge():
    d = Doc(); c = Case("en"); p = c.pat
    sc = scen_en(c.g)
    dr1, dr2 = c.staff(), c.staff()
    addr, _ = addr_en(c.loc, "in" if opt(0.7) else "none")
    adm = c.ref - timedelta(days=ri(2, 12))
    lay = ch(["kv", "kv", "prose", "caps"])
    d.t(fac_en(c.loc, "big"), "\n")
    if opt(0.5):
        d.t(ch(["Department of Medicine", "Dept. of General Medicine", "Medicine Unit III",
                "Department of Obstetrics & Gynaecology" if c.g == "F" else "Department of Orthopaedics"]), "\n")
    d.t(ch(["DISCHARGE SUMMARY", "Discharge Summary", "DISCHARGE CARD", "Dischrage Summary"]), "\n\n")
    if lay in ("kv", "caps"):
        if lay == "caps":
            d.case = "upper"
        F = [(ch(["UHID", "UHID No.", "CR No.", "Hospital Reg. No."]), [E(uhid() if opt(0.7) else cr_no(), "MRN")]),
             (ch(["IPD No.", "IP No.", "Adm. No.", "IPD Reg No"]), [E(ipd_no(), "MRN")]),
             (ch(L_NAME), [hon_en(p), P(p, style=nst())]),
             (ch(["Age/Sex", "Age / Gender", "Age-Sex"]), [agesex_en(c)]),
             guardian_field_en(c),
             (ch(L_ADDR), [E(addr, "ADDRESS")]),
             (ch(L_MOB), [E(phone_en(), "PHONE")])]
        if opt(0.4):
            F.append((ch(L_DOB), [E(fdate(c.dob), "DOB")]))
        if opt(0.45):
            F.append(aadhaar_field())
        if opt(0.35):
            F += abha_fields(c)
        if opt(0.45):
            F.append((ch(["PAN (billing)", "PAN of payer", "PAN No.", "KYC - PAN"]), [E(pan_no(p), "PAN")]))
        if opt(0.2):
            F.append((ch(L_EMAIL_LAB), [E(email_for(p), "EMAIL")]))
        F.append((ch(["Bed No.", "Bed"]), [f"{ri(1, 40)}, {ch(ADULT_WARDS)}"]))
        F.append((ch(["DOA", "Date of Admission", "Adm. Date"]), [fdate(adm)]))
        F.append((ch(["Date of Discharge", "Disch. Date", "DOD"]), [fdate(c.ref)]))
        F.append((ch(["Consultant", "Unit Head", "Treating Doctor"]), [dr_en(), P(dr1, ch(["full", "init"]))]))
        kv(d, F)
        d.case = None
    else:
        d.t(hon_en(p), P(p), f", {c.age}-year-old {'male' if c.g == 'M' else 'female'}, ", guardian_inline_en(c),
            ", resident of ", E(addr, "ADDRESS"), f", was admitted on {fdate(adm)} under ", dr_en(), P(dr1),
            " (UHID ", E(uhid(), "MRN"), f"; Bed {ri(1, 40)}, {ch(ADULT_WARDS)}) and discharged on {fdate(c.ref)}. Contact: ",
            E(phone_en(), "PHONE"), ".\n")
        if opt(0.4):
            d.t("Date of birth as per Aadhaar: ", E(fdate(c.dob), "DOB"), "; Aadhaar ", E(fmt_aadhaar(), "AADHAAR"), ".\n")
    d.t("\n", ch(["Diagnosis: ", "Final Diagnosis: ", "DIAGNOSIS - "]), sc["dx"], "\n\n")
    pre, a, form, rk = att_phrase_en(c)
    his = "his" if c.g == "M" else "her"
    d.t(ch(["History: ", "Presenting complaints: ", "HOPI: "]), P(p, "short"),
        f", {c.age} {ch(['yrs', 'years', 'Y'])}, presented with {sc['cc']}. ")
    d.t(f"{ch(['Patient was', 'Was'])} {pre}", P(a, form), ". ")
    if opt(0.4):
        d.t(f"Past history of {ch(['T2DM x 6 yrs', 'HTN on irregular treatment', 'no major illness', 'pulmonary Koch\'s treated in 2019', 'hypothyroidism on Tab. Thyroxine 50 mcg'])}. ")
    d.t("\n\nOn examination: ", sc["exam"], ".\n\n")
    d.t(ch(["Investigations:", "Investigations (significant):", "INVESTIGATIONS"]), "\n")
    for l in sc["labs"]:
        d.t(ch(["- ", "* ", "  "]), l, "\n")
    d.t("\nCourse in hospital: ", sc["course"], " ")
    d.t(P(p, "short") if opt(0.6) else "Patient", f" is being discharged in {ch(['stable', 'haemodynamically stable', 'satisfactory'])} condition.\n\n")
    d.t(ch(["Treatment on discharge:", "Advice on discharge:", "Rx at discharge:"]), "\n")
    for i, r in enumerate(sc["rx"], 1):
        d.t(f"{i}. {r}\n")
    fu = c.ref + timedelta(days=ri(5, 15))
    d.t(f"\nFollow-up: review in {ch(['Medicine', 'Surgery', 'Ortho'] + (['ANC'] if c.g == 'F' and c.age < 40 else []))} OPD on {fdate(fu)} with reports. ")
    d.t(ch(["In case of emergency call 108.", "Return immediately if fever/bleeding/breathlessness.", "Bring this summary at every visit."]), "\n")
    if opt(0.4):
        d.t(neg_number(), "\n")
    d.t("\n", ch(["Prepared by: ", "Summary prepared by "]), dr_en(), P(dr2, ch(["full", "init"])), ch([" (JR-2)", " (SR)", ", Resident"]))
    d.t("\n", ch(["Checked by: ", "Consultant: ", "Signature of consultant - "]), dr_en(), P(dr1, "last"), "\n")
    return "discharge_summary", d


def en_opd():
    d = Doc(); c = Case("en"); p = c.pat
    sc = scen_en(c.g)
    dr = c.staff()
    lay = ch(["slip", "slip", "sloppy", "clinic"])
    if lay == "slip":
        d.t(fac_en(c.loc), "\n", ch(["OPD SLIP", "Out-Patient Ticket", "OPD Card", "O.P.D. Registration"]), "\n")
        F = [(ch(["OPD Reg. No.", "OPD No.", "Reg. No.", "OP No."]), [E(opd_no(), "MRN")]), ("Date", [fdate(c.ref)]),
             (ch(L_NAME), [hon_en(p), P(p, style=nst())]), (ch(["Age/Sex", "Age", "Age/Gender"]), [agesex_en(c)])]
        if opt(0.5):
            F.append(guardian_field_en(c))
        addr, pin = addr_en(c.loc, ch(["in", "none", "sep"]), short=opt(0.5))
        F.append((ch(L_ADDR), [E(addr, "ADDRESS")]))
        if addr and pin and not addr.endswith(pin) and opt(0.5):
            F.append((ch(["PIN", "Pin code", "Pincode"]), [E(pin, "PINCODE")]))
        F.append((ch(L_MOB), [E(phone_en(), "PHONE")]))
        if opt(0.25):
            F.append(aadhaar_field())
        if opt(0.3):
            F += abha_fields(c)
        F.append(("Dept.", [ch(["General Medicine", "Medicine", "OBG", "Surgery", "NCD Clinic", "Chest & TB"]) + f", Room No. {ri(1, 30)}"]))
        kv(d, F)
        d.t("\n", ch(["C/O: ", "c/o ", "Complaints: ", "Chief complaints - "]), sc["cc"], "\n")
        d.t(ch(["O/E: ", "OE- ", "Examination: "]), sc["exam"], "\n")
        d.t(ch(["Rx", "Adv:", "Treatment:"]), "\n")
        for i, r in enumerate(sc["rx"], 1):
            d.t(f"{i}. {r}\n")
        if opt(0.7):
            d.t(ch(["Inv: ", "Investigations advised: "]), ", ".join(sc["inv"]), "\n")
        d.t(ch(["Review after 3 days", "Review SOS", "Follow up after 1 week with reports",
                f"Review on {fdate(c.ref + timedelta(days=ri(3, 14)))}"]), ".\n")
        d.t(ch(["", "Sign: ", "Signature of MO: "]), dr_en(), P(dr, ch(["full", "init"])), ch([" (MO)", ", Medical Officer", " MBBS", ""]), "\n")
    elif lay == "sloppy":
        d.case = "lower"
        pre, a, form, rk = att_phrase_en(c)
        addr, _ = addr_en(c.loc, ch(["in", "none"]), short=True)
        d.t(f"opd {fdate(c.ref, 'dmy/y')} ", ch(["reg ", "opd no ", "op "]), E(opd_no(), "MRN"), "\n")
        d.t("pt ", P(p), f" {agesex_en(c)} ", ch(["r/o ", "from ", "add "]), E(addr, "ADDRESS"), "\n")
        d.t(ch(["mob ", "ph ", "cont "]), E(fmt_mobile(ch(["plain", "split", "zero"])), "PHONE"), "  ", pre, P(a, form), "\n")
        d.t("c/o ", sc["cc"], "\n", "o/e ", sc["exam"], "\n", "adv\n")
        for r in sc["rx"]:
            d.t("- ", r, "\n")
        d.t(ch(["inv ", "tests "]), " ".join(sc["inv"]), "\n")
        if opt(0.5):
            d.t(f"{rk} told that ", P(p, "short"), f" stopped meds {ri(2, 10)} days back\n")
        d.t("rv sos / after ", str(ri(3, 7)), " days\n")
        d.case = None
        d.t("Dr ", P(dr, "last"), "\n")
    else:
        d.t(dr_en(), P(dr), "\n", ch(["MBBS, MD (Medicine)", "MBBS, DGO", "MBBS, MS (Gen Surgery)", "MBBS, DNB (Family Medicine)", "MBBS, MD (Paed)"]),
            "\n", ch(["Consultant Physician", "Family Physician", "Visiting Consultant"]), "\n")
        d.t(f"{ch(['Arogya', 'Sanjeevani', 'Shanti', 'Jeevan Jyoti', 'Care'])} Clinic, {ch(['Station Road', 'Main Bazaar', 'Civil Lines', 'Gandhi Chowk', 'Nehru Market'])}, {c.loc.district}\n")
        d.t(f"Timings: {ri(9, 11)} am - 1 pm, {ri(5, 6)} - 8 pm (Sunday closed)\n", "-" * ri(20, 40), "\n")
        d.t(f"Date: {fdate(c.ref)}\n", "Patient: ", hon_en(p), P(p, style=nst()), f"   {agesex_en(c)}\n")
        if opt(0.6):
            d.t(ch(L_MOB), ": ", E(fmt_mobile(), "PHONE"), "\n")
        if opt(0.4):
            d.t(ch(L_ADDR), ": ", E(addr_en(c.loc, "none", short=True)[0], "ADDRESS"), "\n")
        if opt(0.5):
            d.t(ch(L_EMAIL_LAB), ": ", E(email_for(p), "EMAIL"), "\n")
        if opt(0.3):
            d.t("PAN (for receipt): ", E(pan_no(p), "PAN"), "\n")
        d.t("\n", sc["cc"].capitalize(), ".\n", sc["exam"], ".\n\nRx\n")
        for i, r in enumerate(sc["rx"], 1):
            d.t(f"{i}) {r}\n")
        d.t(f"\nAdv: {', '.join(sc['inv'])}. Review after {ri(5, 15)} days.\n")
        if opt(0.5):
            d.t("Fee received Rs ", str(ch([200, 300, 400, 500])), " (", neg_number(), ")\n")
    return "opd_prescription", d


L_EMAIL_LAB = ["Email", "E-mail", "email id", "Email ID"]


def en_referral():
    d = Doc(); c = Case("en"); p = c.pat
    sc = scen_en(c.g)
    mo = c.staff()
    src = fac_en(c.loc, "small")
    dst = fac_en(c.loc, "big")
    lay = ch(["letter", "letter", "slip"])
    addr, pin = addr_en(c.loc, ch(["in", "in", "sep", "none"]))
    if lay == "letter":
        d.t(f"Date: {fdate(c.ref)}\n\nTo,\nThe {ch(['Medical Superintendent', 'Chief Medical Officer', 'Physician on duty', 'HOD Medicine'])}\n{dst}\n\n")
        d.t("Sub: Referral of ", hon_en(p, prob=0.9), P(p), "\n\n", ch(["Respected Sir/Madam,", "Dear Colleague,", "Sir,"]), "\n")
        d.t("I am referring ", hon_en(p), P(p, style=nst()), f", aged {c.age} years, ", guardian_inline_en(c), ", resident of ",
            E(addr, "ADDRESS"))
        if pin and not addr.endswith(pin) and opt(0.6):
            d.t(" (PIN ", E(pin, "PINCODE"), ")")
        d.t(" (", ch(["OPD No. ", "Reg. No. ", "MRD No. "]), E(ch([opd_no(), mrd_no()]), "MRN"), f"), who presented to {src} on {fdate(c.ref)} with {sc['cc']}. ")
        d.t(f"On examination: {sc['exam']}. ")
        d.t(f"{ch(['Provisional diagnosis', 'Impression'])}: {sc['dx']}.\n\n")
        d.t(f"Treatment given here: {sc['rx'][0]}; {sc['rx'][-1]}. ")
        d.t(f"Patient is being referred for {ch(['further management', 'specialist opinion and admission', 'higher centre care', 'ICU care'])} as facilities are not available at this centre. ")
        pre, a, form, rk = att_phrase_en(c)
        his = "He" if c.g == "M" else "She"
        d.t(f"{his} is accompanied by {rk} ", P(a, form), " (Mob. ", E(fmt_mobile(), "PHONE"))
        if opt(0.25):
            d.t(", res. ", E(landline(), "PHONE"))
        if opt(0.3):
            d.t(", e-mail ", E(email_for(a), "EMAIL"))
        d.t("). ")
        if opt(0.5):
            d.t("Aadhaar of patient: ", E(fmt_aadhaar(), "AADHAAR"), ". ")
        d.t(ch(["Referred via 108 ambulance.", "Shifted in 102 vehicle.", "Patient going by own vehicle."]), "\n\n")
        d.t(ch(["Thanking you,", "Kindly do the needful.", "Regards,"]), "\n\n", dr_en(), P(mo), "\n", ch(["Medical Officer", "MO i/c", "Medical Officer In-charge"]), f", {src}\n")
        if opt(0.4):
            d.t(ch(["Copy to: ", "CC: "]), "ASHA ", P(c.staff("F", pool=c.pool), ch(["full", "short"])), " for follow-up\n")
    else:
        d.t(ch(["REFERRAL SLIP", "Referral Card", "REFERRAL FORM (Higher Facility)"]), "\n")
        F = [("From", [src]), ("To", [dst]), ("Date/Time", [f"{fdate(c.ref)} {ri(1, 12)}:{ri(0, 59):02d} {ch(['AM', 'PM'])}"]),
             (ch(L_NAME), [hon_en(p), P(p, style=nst())]), ("Age/Sex", [agesex_en(c)]), guardian_field_en(c),
             (ch(L_ADDR), [E(addr, "ADDRESS")])]
        if pin and not addr.endswith(pin):
            F.append(("PIN", [E(pin, "PINCODE")]))
        F += [(ch(L_MOB), [E(phone_en(), "PHONE")]), ("Reason for referral", [sc["dx"]]),
              ("Vitals at referral", [sc["exam"]]), ("Treatment given", ["; ".join(sc["rx"][:2])])]
        if opt(0.6):
            F.append(idproof_field(p))
        kv(d, F)
        anm = c.staff("F", pool=c.pool)
        d.t(ch(["Referred by: ", "Escorted by: "]), ch(["ANM ", "Staff Nurse ", "SN ", "CHO "]), P(anm, ch(["full", "short"])), ", ", dr_en(), P(mo, "init"), " (MO)\n")
        d.t("Ambulance: 108, ", f"Vehicle reached at {ri(1, 12)}:{ri(0, 59):02d}. {neg_number()}\n")
    return "referral_letter", d


def en_asha():
    d = Doc(); c = Case("en", g="F", age=(19, 34))
    p = c.pat
    asha = c.staff("F", pool=c.pool)
    anm = c.staff("F")
    kind, hus = ("husband", relative(p, "M"))
    c._rel = (kind, hus)
    lay = ch(["hbnc", "hbnc", "ncd", "prose"])
    addr, pin = addr_en(c.loc, ch(["in", "none", "sep"]))
    if lay == "hbnc":
        d.t(ch(["ASHA HOME VISIT REPORT - HBNC", "Home Based Newborn Care - Visit Form", "HBNC visit format"]), "\n")
        F = [("ASHA", [P(asha, style=nst())]), ("Sub-centre", [f"{c.loc.po}"]), ("Block", [c.loc.block]),
             ("Visit date", [fdate(c.ref)]), ("Visit no.", [ch(["1st (Day 3)", "2nd (Day 7)", "3rd (Day 14)", "4th (Day 21)"])]),
             (ch(["Mother", "Name of mother", "Mother's name"]), [P(p)]), (ch(["W/o", "Husband"]), [P(hus)]),
             (ch(["Household address", "Address", "House"]), [E(addr, "ADDRESS")])]
        if pin and not addr.endswith(pin):
            F.append(("Pin", [E(pin, "PINCODE")]))
        F.append((ch(L_MOB), [E(fmt_mobile(), "PHONE")]))
        bdob = c.ref - timedelta(days=ri(3, 21))
        F += [("Baby", [ch(["Baby of ", "B/o "]), P(p, ch(["full", "short"])), f", {ch(['male', 'female'])}"]),
              ("Baby DOB", [E(fdate(bdob), "DOB")]), ("Birth weight", [f"{R.uniform(1.9, 3.4):.2f} kg"]),
              ("Place of delivery", [fac_en(c.loc, "small")])]
        if opt(0.5):
            F.append(("MCP card / RCH ID", [E(rch_id(), "MRN")]))
        kv(d, F, pair=0.2)
        d.t("\nFindings: ", ch(["baby feeding well, temp normal (98.4 F), cord dry", "baby not sucking well, temp 99.8 F, weight 2.1 kg",
                                "jaundice up to chest, referred", "eyes sticky, cord clean"]), ". ")
        d.t("Mother: ", ch(["no excessive bleeding, Hb not checked", f"Hb {hb()} g/dL, taking IFA", "fever since 2 days, advised to visit PHC"]), ".\n")
        d.t("Counselling: exclusive breastfeeding, KMC, keep baby warm. Next visit on ", fdate(c.ref + timedelta(days=7)), ".\n")
        d.t("Incentive claimed: Rs 250 (HBNC). ", ch(["", "Copy shared with ANM "]))
        if d.buf[-1].endswith("ANM "):
            d.t(P(anm))
        d.t("\n")
    elif lay == "ncd":
        head = make_person(c.pool, "M", family=p.family)
        old = relative(head, "F")
        d.t(ch(["ASHA visit - NCD / CBAC follow-up", "Household visit note (CBAC)"]), f"\nDate: {fdate(c.ref)}\n")
        d.t(f"Visited household of {ch(['Shri ', 'Mr. ', ''])}", P(head), " at ", E(addr, "ADDRESS"), ". ")
        if pin and not addr.endswith(pin):
            d.t("Pincode ", E(pin, "PINCODE"), ". ")
        d.t(f"His mother ", P(old, ch(["full", "short"])), f", {ri(60, 80)} yrs, k/c/o HTN, not taking Tab. Amlodipine 5 mg since {ri(5, 20)} days. ")
        d.t(f"BP today {ri(150, 190)}/{ri(90, 110)} mmHg, RBS {ri(110, 260)} mg/dL (glucometer). CBAC score {ri(4, 9)}. ")
        d.t("Daughter-in-law ", P(p, "short"), f" ({c.age} yrs) also screened - BP {ri(100, 128)}/{ri(64, 84)}, Hb {hb()} g/dL. ")
        d.t("Family contact: ", E(phone_en(), "PHONE"), ".\n")
        d.t("Advised to attend NCD clinic at ", fac_en(c.loc, "small"), " on Thursday. ")
        d.t("Reported by ASHA ", P(asha), ch(["", ", verified by ANM "]))
        if d.buf[-1].endswith("ANM "):
            d.t(P(anm, "short"))
        d.t(".\n")
    else:
        d.case = ch([None, "lower"])
        d.t(f"asha report {fdate(c.ref, 'dmy/y')}\n" if d.case else f"ASHA report {fdate(c.ref)}\n")
        d.t("visited ", P(p), f" w/o ", P(hus), ", ", E(addr, "ADDRESS"), ". ")
        d.t(f"she is {ri(6, 9)} months pregnant, {ch(['3rd', '4th', '2nd'])} ANC due. her brother ", P(relative(p, "M"), "short"),
            " brought her to the VHSND session last month. ")
        d.t(f"complains of pedal oedema and headache, BP {ri(140, 160)}/{ri(90, 104)}. ")
        d.t("husband phone ", E(fmt_mobile(ch(["plain", "dash", "zero"])), "PHONE"), ". ")
        d.t("advised immediate visit to ", fac_en(c.loc, "small"), ". mcts/rch no ", E(rch_id(), "MRN"), ".\n")
        d.t("asha: ", P(asha), "\n")
        d.case = None
    return "asha_home_visit", d


def en_lab():
    d = Doc(); c = Case("en"); p = c.pat
    ref = c.staff()
    path = c.staff()
    tech = c.staff(pool=c.pool)
    lab = ch([f"District Hospital Laboratory, {c.loc.district}", f"{ch(['City', 'Precision', 'Accurate', 'Life Care', 'Shree Sai'])} Diagnostics",
              f"Central Pathology Lab, {fac_en(c.loc, 'big')}", f"IDSP District Public Health Lab, {c.loc.district}"])
    caps = opt(0.45)
    d.t(lab, "\n")
    if opt(0.5):
        d.t(f"{ch(['Station Road', 'Civil Lines', 'Opp. Collectorate', 'Near Clock Tower'])}, {c.loc.district}   |   NABL Cert. No. MC-{ri(1000, 7999)}\n")
    d.t("=" * ri(30, 50), "\n")
    if caps:
        d.case = "upper"
    F = [(ch(["Patient Name", "Name", "PATIENT"]), [hon_en(p), P(p)]), (ch(["Age/Sex", "Age / Sex"]), [agesex_en(c)]),
         (ch(["Ref. By", "Referred by", "Ref. Doctor"]), [dr_en(), P(ref, ch(["init", "full"]))]),
         (ch(["Sample ID", "Lab No.", "Sample No.", "Barcode"]), [E(lab_id(), "MRN")]),
         (ch(["UHID", "Patient ID", "Reg. No.", "MR No."]), [E(ch([uhid(), mrd_no(), cr_no()]), "MRN")]),
         ("Collected", [f"{fdate(c.ref)} {ri(7, 11)}:{ri(0, 59):02d} AM"]),
         ("Reported", [f"{fdate(c.ref + timedelta(days=ri(0, 1)))} {ri(1, 6)}:{ri(0, 59):02d} PM"])]
    if opt(0.4):
        F.append((ch(L_MOB), [E(phone_en(), "PHONE")]))
    if opt(0.25):
        F.append((ch(L_DOB), [E(fdate(c.dob), "DOB")]))
    if opt(0.3):
        F += abha_fields(c)
    if opt(0.4):
        F.append((ch(L_EMAIL_LAB), [E(email_for(p), "EMAIL")]))
    if opt(0.2):
        F.append(("Address", [E(addr_en(c.loc, "in", short=True)[0], "ADDRESS")]))
    F.append(("Bed", [f"{ri(1, 40)} / {ch(WARDS)}" if opt(0.5) else "OPD"]))
    kv(d, F, sep=ch([" : ", ": ", "\t: "]), pair=0.6)
    d.case = None
    d.t("-" * ri(30, 50), "\n")
    d.t(ch(["TEST                 RESULT    UNIT     BIO. REF. INTERVAL", "Investigation | Result | Units | Reference range",
            "Test Name\tValue\tUnit\tNormal"]), "\n")
    for name, val, unit, rng in lab_panel():
        d.t(f"{name:<22} {val:<9} {unit:<9} {rng}\n")
    d.t("\n", ch(["Kit lot / QC: ", "Reagent: "]), neg_number(), "\n")
    d.t(ch(["Checked by: ", "Technician: ", "Lab Tech: "]), P(tech, ch(["short", "full"])), "    ",
        ch(["Pathologist: ", "Verified by: "]), dr_en(), P(path), ch([", MD (Path)", " (DCP)", ""]), "\n")
    d.t("*** End of report ***\n")
    return "lab_report_header", d


def en_anc():
    d = Doc(); c = Case("en", g="F", age=(18, 36)); p = c.pat
    kind, hus = ("husband", relative(p, "M")); c._rel = (kind, hus)
    asha, anm = c.staff("F", pool=c.pool), c.staff("F")
    lmp = c.ref - timedelta(days=ri(60, 250))
    edd = lmp + timedelta(days=280)
    addr, pin = addr_en(c.loc, ch(["in", "in", "sep", "none"]))
    d.t(ch(["MOTHER AND CHILD PROTECTION CARD (ANC section)", "ANC Register extract", "Antenatal Card"]), "\n")
    d.t(f"Facility: {fac_en(c.loc, 'small')}\n")
    F = [(ch(["RCH ID", "RCH Reg. No.", "MCTS ID"]), [E(rch_id(), "MRN")]),
         (ch(["Name of PW", "Name of woman", "Name", "Beneficiary"]), [hon_en(p, prob=0.4), P(p, style=nst())]),
         ("Age", [f"{c.age} yrs"]), (ch(["Husband's name", "W/o", "Husband"]), [P(hus)]),
         (ch(L_ADDR), [E(addr, "ADDRESS")])]
    if pin and not addr.endswith(pin):
        F.append(("PIN", [E(pin, "PINCODE")]))
    F.append((ch(L_MOB), [E(fmt_mobile(), "PHONE")]))
    if opt(0.5):
        F.append(("Husband mobile", [E(fmt_mobile(), "PHONE")]))
    if opt(0.55):
        F.append((ch(["Aadhaar (woman)", "Aadhaar", "Aadhaar No."]), [E(fmt_aadhaar(), "AADHAAR")]))
    if opt(0.3):
        F.append((ch(L_DOB), [E(fdate(c.dob), "DOB")]))
    if opt(0.4):
        F += abha_fields(c)
    if opt(0.2):
        F.append((ch(["Voter ID", "EPIC No."]), [E(epic_no(), "VOTER_ID")]))
    if opt(0.2):
        F.append((ch(L_EMAIL_LAB), [E(email_for(p), "EMAIL")]))
    F += [("LMP", [fdate(lmp)]), ("EDD", [fdate(edd)]),
          ("Obstetric", [f"G{ri(1, 4)} P{ri(0, 3)} L{ri(0, 3)} A{ri(0, 1)}"]), ("Blood group", [ch(["B +ve", "O +ve", "A +ve", "AB +ve", "O -ve"])])]
    kv(d, F)
    d.t("\nANC visits:\n", ch(["Date | Wt (kg) | BP | Hb | FH (cm) | FHR", "Date  Wt  BP  Hb  FH  FHR"]), "\n")
    dt = lmp + timedelta(days=ri(60, 90))
    for _ in range(ri(2, 4)):
        if dt > c.ref:
            break
        d.t(f"{fdate(dt, 'dmy/')} | {R.uniform(42, 68):.1f} | {ri(100, 150)}/{ri(64, 98)} | {hb()} | {ri(12, 34)} | {ri(128, 156)}\n")
        dt += timedelta(days=ri(28, 45))
    d.t(f"\nTd-1 given {fdate(lmp + timedelta(days=ri(90, 110)))} ({neg_number()}), Td-2 after 4 weeks.\n")
    d.t(f"IFA tabs issued: {ch([90, 180, 360])}; Calcium 500 mg BD x {ri(60, 180)} days; Albendazole 400 mg once in 2nd trimester.\n")
    d.t("High risk: ", ch(["Nil", f"Severe anaemia (Hb {R.uniform(5.5, 6.9):.1f})", "Previous LSCS", "PIH", "Short stature (< 140 cm)"]), "\n")
    d.t("Birth plan: ", fac_en(c.loc, "big"), "; JSY eligible: ", ch(["Yes", "No"]), "\n")
    d.t("ASHA: ", P(asha), " (", E(fmt_mobile(), "PHONE"), ")    ANM: ", P(anm, ch(["full", "short"])), "\n")
    if opt(0.5):
        d.t("Note: ", P(p, "short"), " came with her mother-in-law ", P(relative(hus, "F"), "short"), "; counselled on danger signs.\n")
    return "anc_card", d


def en_tb():
    d = Doc(); c = Case("en"); p = c.pat
    sup = c.staff(pool=c.pool)
    mo = c.staff()
    wife_or_h = relative(p, "F" if c.g == "M" else "M")
    child = relative(p, ch("MF"))
    start = c.ref - timedelta(days=ri(56, 170))
    w0 = ri(34, 50)
    addr, pin = addr_en(c.loc, ch(["in", "sep", "none"]))
    lay = ch(["kv", "kv", "prose"])
    d.t(ch(["NTEP - TREATMENT FOLLOW-UP (Ni-kshay)", "TB Follow-up Card", "Nikshay follow-up visit"]), "\n")
    d.t(f"TU: {c.loc.block}   DMC/PHI: {fac_en(c.loc, 'small')}\n")
    if lay == "kv":
        F = [(ch(["Nikshay ID", "Ni-kshay ID", "NIKSHAY ID"]), [E(nikshay_id(), "MRN")]), (ch(["TB No.", "TB Reg. No."]), [E(tb_no(), "MRN")]),
             (ch(L_NAME), [hon_en(p), P(p, style=nst())]), ("Age/Sex", [agesex_en(c)]), (ch(L_ADDR), [E(addr, "ADDRESS")])]
        if pin and not addr.endswith(pin):
            F.append(("PIN", [E(pin, "PINCODE")]))
        F.append((ch(L_MOB), [E(fmt_mobile(), "PHONE")]))
        if opt(0.4):
            F.append(aadhaar_field())
        if opt(0.3):
            F += abha_fields(c)
        if opt(0.3):
            F.append(idproof_field(p))
        F += [("Type", ["New, pulmonary, microbiologically confirmed"]), ("Regimen", [f"2HRZE/4HRE, weight band {ch(['30-39', '39-54', '55-69'])} kg"]),
              ("Treatment start", [fdate(start)]),
              ("Treatment supporter", [P(sup), ch([" (ASHA)", " (AWW)", " (community volunteer)"]), ", ", E(fmt_mobile(), "PHONE")])]
        kv(d, F)
    else:
        d.t("Ni-kshay ID ", E(nikshay_id(), "MRN"), ". ", hon_en(p), P(p), f", {c.age} {'M' if c.g == 'M' else 'F'}, ", E(addr, "ADDRESS"))
        if pin and not addr.endswith(pin):
            d.t(", pincode ", E(pin, "PINCODE"))
        d.t(", contact ", E(fmt_mobile(), "PHONE"), f". Started on 2HRZE on {fdate(start)}. Treatment supporter ", P(sup), ".\n")
    d.t(f"\nFollow-up visit ({ch(['end of IP', 'month 2', 'month 4', 'month 5', 'end of treatment'])}), {fdate(c.ref)}:\n")
    d.t(f"- Weight {w0} -> {w0 + ri(1, 6)} kg; sputum smear {ch(['negative', 'negative', 'scanty positive'])}; LFT: SGPT {ri(18, 70)} IU/L\n")
    d.t(f"- Adherence {ri(86, 100)}% (99DOTS / MERM); missed doses {ri(0, 6)}\n")
    d.t(f"- ADR: {ch(['nil', 'mild joint pain (Tab. Paracetamol)', 'nausea, managed', 'tingling feet, pyridoxine continued'])}\n")
    d.t(f"- Ni-kshay Poshan Yojana: DBT Rs {ch(['1,000', '1,000', '500'])} credited for {ch(['Sept', 'Aug', 'Oct'])}\n")
    d.t("- Contact tracing: ", ch(["wife ", "spouse "]) if c.g == "M" else "husband ", P(wife_or_h, "short"),
        f" screened, symptom negative; {ch(['son', 'daughter'])} ", P(child, "short"), f" ({ri(2, 9)} yrs) started on TPT (3HP).\n")
    d.t("Next visit: ", fdate(c.ref + timedelta(days=30)), ". MO: ", dr_en(), P(mo, ch(["full", "init", "last"])), "\n")
    return "tb_nikshay_followup", d


VACC = [("BCG", "at birth"), ("OPV-0", "at birth"), ("Hep B birth dose", "at birth"), ("OPV-1", "6 wk"), ("Pentavalent-1", "6 wk"),
        ("Rota-1", "6 wk"), ("fIPV-1", "6 wk"), ("PCV-1", "6 wk"), ("OPV-2", "10 wk"), ("Pentavalent-2", "10 wk"),
        ("Rota-2", "10 wk"), ("OPV-3", "14 wk"), ("Pentavalent-3", "14 wk"), ("fIPV-2", "14 wk"), ("MR-1", "9 mo"),
        ("JE-1", "9 mo"), ("Vit A (1st)", "9 mo")]


def en_imm():
    d = Doc(); c = Case("en", age=0); child = c.pat
    mother = make_person(c.pool, "F", family=child.family)
    father = relative(child, "M")
    anm = c.staff("F")
    bdob = c.ref - timedelta(days=ri(45, 300))
    addr, pin = addr_en(c.loc, ch(["in", "none", "sep"]))
    named = opt(0.6)
    d.t(ch(["IMMUNISATION RECORD", "Child Immunisation Card (U-WIN extract)", "Routine Immunisation - Due list entry"]), "\n")
    F = [(ch(["RCH ID", "U-WIN Reg. No.", "Child ID"]), [E(rch_id(), "MRN")])]
    if named:
        F.append((ch(["Child's name", "Name of child", "Child"]), [P(child, ch(["short", "full"]))]))
    else:
        F.append((ch(["Child's name", "Name of child"]), [ch(["Baby of ", "B/o ", "BO "]), P(mother)]))
    F += [("Sex", [ch(["M", "Male"]) if c.g == "M" else ch(["F", "Female"])]), (ch(["DOB", "Date of birth", "D.O.B"]), [E(fdate(bdob), "DOB")]),
          ("Birth weight", [f"{R.uniform(1.8, 3.6):.2f} kg"]), (ch(["Mother", "Mother's name"]), [hon_en(mother, prob=0.3), P(mother)]),
          (ch(["Father", "Father's name"]), [P(father)]), (ch(L_ADDR), [E(addr, "ADDRESS")])]
    if pin and not addr.endswith(pin):
        F.append(("Pin code", [E(pin, "PINCODE")]))
    F.append((ch(L_MOB), [E(fmt_mobile(), "PHONE")]))
    if opt(0.25):
        F.append(("Mother's Aadhaar", [E(fmt_aadhaar(), "AADHAAR")]))
    if opt(0.25):
        F.append(("Child ABHA", [E(abha_number(), "ABHA_NUMBER")]))
    kv(d, F)
    d.t(f"Session site: {ch(['AWC', 'Anganwadi Centre', 'Sub-centre'])} {c.loc.po}\n\n")
    d.t(ch(["Vaccine | Due | Given on | Batch", "Vaccine\tAge\tDate given\tBatch No."]), "\n")
    age_days = (c.ref - bdob).days
    for v, due in VACC:
        off = {"at birth": 0, "6 wk": 42, "10 wk": 70, "14 wk": 98, "9 mo": 270}[due]
        if off > age_days:
            break
        given = bdob + timedelta(days=off + ri(0, 9))
        d.t(f"{v} | {due} | {fdate(given, 'dmy/')} | {ch('ABCDEKLMS')}{ri(2300, 2699)}{ri(100, 999)}\n")
    d.t(f"\nAEFI: {ch(['none', 'mild fever after Penta-1, settled', 'none reported'])}. Wt today {R.uniform(3.2, 9.4):.1f} kg.\n")
    d.t("Vaccinator: ANM ", P(anm), "\n")
    if opt(0.5):
        d.t("Due list given to ASHA ", P(c.staff("F", pool=c.pool), "short"), "; mother ", P(mother, "short"), " informed by phone.\n")
    return "immunisation_record", d


def en_death():
    d = Doc(); c = Case("en", age=(45, 92)); p = c.pat
    inf_kind = ch(["son", "daughter", "brother", "wife" if c.g == "M" else "husband", "grandson"])
    inf = relative(p, "F" if inf_kind in ("daughter", "wife") else "M")
    other = relative(p, "M")
    drc = c.staff()
    tod = c.ref
    addr, pin = addr_en(c.loc, ch(["in", "in", "none"]))
    sc = scen_en(c.g)
    d.t(ch(["DRAFT - Death notification (Form 2) / MCCD Form 4", "Medical Certificate of Cause of Death - DRAFT", "DEATH INTIMATION (draft for CRS)"]), "\n")
    d.t(f"Hospital: {fac_en(c.loc, 'big')}   Ward/Bed: {ch(ADULT_WARDS)} / {ri(1, 30)}\n")
    F = [(ch(["Name of deceased", "Deceased", "Name"]), [ch(["Late ", "Late Shri " if c.g == "M" else "Late Smt. ", ""]), P(p, style=nst())]),
         ("Sex", ["Male" if c.g == "M" else "Female"]), ("Age at death", [f"{c.age} years"])]
    if opt(0.5):
        F.append((ch(["DOB (if known)", "Date of birth"]), [E(fdate(c.dob), "DOB")]))
    F.append(guardian_field_en(c))
    F.append((ch(["Permanent address", "Usual residence", "Address"]), [E(addr, "ADDRESS")]))
    F += [("Date & time of death", [f"{fdate(tod)} {ri(1, 12)}:{ri(0, 59):02d} {ch(['AM', 'PM'])}"]),
          ("Admission no.", [E(ipd_no(), "MRN")])]
    kv(d, F)
    d.t("\nCause of death:\n", f"(a) Immediate: {ch(['Septic shock', 'Acute respiratory failure', 'Hypovolemic shock', 'Hepatic encephalopathy'])}\n")
    d.t(f"(b) Antecedent: {sc['dx']}\n(c) Underlying: {ch(['Type 2 diabetes mellitus', 'Chronic kidney disease stage 5', 'Alcoholic liver disease', 'Hypertension'])}\n")
    d.t("Manner: natural. Interval onset to death: ", f"{ri(2, 20)} days\n\n")
    d.t("Informant: ", P(inf), f" ({inf_kind})", ", ", ch(L_MOB), " ", E(phone_en(), "PHONE"))
    if opt(0.75):
        lab, val = idproof_field(inf)
        d.t(", ", lab, " ", val)
    d.t("\n")
    if opt(0.6):
        d.t(f"Body handed over to relatives ({ch(['brother', 'nephew', 'son-in-law', 'cousin'])} ", P(other, ch(["short", "full"])),
            f") at {ri(1, 12)}:{ri(0, 59):02d}. ")
    d.t("Certifying doctor: ", dr_en(), P(drc), ch([" (MO on duty)", ", Asst. Professor", ""]), "\n")
    return "death_notification_draft", d


def en_birth():
    d = Doc(); c = Case("en", g="F", age=(19, 36)); m = c.pat
    f = relative(m, "M")
    sn = c.staff("F")
    drc = c.staff()
    bdate = c.ref
    addr, pin = addr_en(c.loc, ch(["in", "in", "sep"]))
    d.t(ch(["BIRTH NOTIFICATION - draft for CRS entry", "Birth Report (Form 1) - DRAFT", "Live birth intimation"]), "\n")
    d.t(f"Place of birth: {ch(['Labour room', 'OT', 'Labour room (MCH wing)'])}, {fac_en(c.loc)}\n")
    F = [(ch(["Date of birth", "DOB", "Date of birth of child"]), [E(fdate(bdate), "DOB")]),
         ("Time", [f"{ri(1, 12)}:{ri(0, 59):02d} {ch(['AM', 'PM'])}"]), ("Sex", [ch(["Male", "Female"])]),
]
    kv(d, F[:3])
    d.t("Name of child: ")
    if opt(0.5):
        d.t(ch(["Baby of ", "B/O "]), P(m), "\n")
    else:
        d.t(ch(["(not named yet)", "--", "to be added later"]), "\n")
    F2 = [("Birth weight", [f"{R.uniform(1.9, 3.8):.2f} kg"]), ("Type of delivery", [ch(["Normal vaginal", "LSCS", "Assisted (vacuum)"])]),
          (ch(["Mother's name", "Name of mother"]), [hon_en(m, prob=0.4), P(m, style=nst())]), ("Mother's age at delivery", [f"{c.age} yrs"]),
          (ch(["Father's name", "Name of father"]), [P(f)])]
    if opt(0.6):
        F2.append(("Mother's Aadhaar", [E(fmt_aadhaar(), "AADHAAR")]))
    if opt(0.55):
        lab, val = idproof_field(f)
        F2.append(("Father - " + lab, val))
    if opt(0.3):
        F2.append(("Mother's ABHA address", [E(abha_address(m, c.dob.year), "ABHA_ADDRESS")]))
    if opt(0.2):
        F2.append(("Father's e-mail", [E(email_for(f), "EMAIL")]))
    F2.append((ch(["Address of parents", "Residence of mother", "Usual residence"]), [E(addr, "ADDRESS")]))
    if pin and not addr.endswith(pin):
        F2.append(("PIN", [E(pin, "PINCODE")]))
    F2.append((ch(L_MOB), [E(fmt_mobile(), "PHONE")]))
    F2 += [("Gestation", [f"{ri(34, 40)} weeks"]), ("APGAR", [f"{ri(6, 8)}/{ri(8, 10)}"]), ("JSY", [ch(["eligible - Rs 1,400", "not eligible", "eligible - Rs 700"])])]
    kv(d, F2)
    d.t("Delivery conducted by: ", ch(["SN ", "Staff Nurse ", "Sister "]), P(sn, ch(["full", "short"])), " under supervision of ", dr_en(), P(drc, ch(["full", "init"])), "\n")
    d.t(f"Vit K 1 mg IM given; BCG/OPV-0/Hep B given {fdate(bdate + timedelta(days=ri(0, 1)))} ({neg_number()}).\n")
    return "birth_notification_draft", d


def en_tele():
    d = Doc(); c = Case("en"); p = c.pat
    sc = scen_en(c.g)
    hubdr, cho = c.staff(), c.staff(pool=c.pool)
    lay = ch(["system", "system", "note"])
    if lay == "system":
        d.t(ch(["eSanjeevani - AB-HWC Teleconsultation", "Teleconsultation e-Prescription", "eSanjeevani OPD consult summary"]), "\n")
        F = [(ch(["Consultation ID", "Consult ID", "Patient ID"]), [E(esj_id(), "MRN")]), ("Date", [fdate(c.ref)]),
             ("Spoke", [f"AB-HWC {c.loc.po} (CHO: ", P(cho), ")"]), ("Hub", [fac_en(c.loc, "big")]),
             ("Hub doctor", [dr_en(), P(hubdr), ch([" (MD Medicine)", " (MBBS)", " (MD Paediatrics)"])]),
             (ch(L_NAME), [P(p, style=nst())]), ("Age/Sex", [agesex_en(c)])]
        F += abha_fields(c, both=opt(0.7))
        F.append((ch(L_MOB), [E(fmt_mobile(), "PHONE")]))
        if opt(0.4):
            F.append((ch(L_ADDR), [E(addr_en(c.loc, ch(["in", "none"]), short=True)[0], "ADDRESS")]))
        if opt(0.5):
            F.append((ch(L_EMAIL_LAB), [E(email_for(p), "EMAIL")]))
        if opt(0.3):
            F.append((ch(L_DOB), [E(fdate(c.dob), "DOB")]))
        kv(d, F)
        d.t("\nChief complaints: ", sc["cc"], "\n", "Vitals (at spoke): ", sc["exam"], "\n", "Provisional diagnosis: ", sc["dx"], "\n")
        d.t("e-Prescription:\n")
        for i, r in enumerate(sc["rx"], 1):
            d.t(f"{i}. {r}\n")
        d.t(f"Advice: {', '.join(sc['inv'])}; follow-up teleconsult after {ri(3, 10)} days.\n")
        d.t("This is a system generated e-prescription.\n")
    else:
        d.case = ch([None, "lower"])
        pre, a, form, rk = att_phrase_en(c)
        d.t(f"Telephonic consult {fdate(c.ref)} {ri(10, 18)}:{ri(0, 59):02d}\n")
        d.t(f"Spoke with patient's {rk} ", P(a, form), " on ", E(phone_en(), "PHONE"), " regarding ", P(p),
            f" ({agesex_en(c)}). ")
        d.t(f"{rk.capitalize()} reports {sc['cc']}. ")
        d.t("Records reviewed under ABHA ", E(abha_number(), "ABHA_NUMBER"), ". ")
        d.t(f"Advised: {sc['rx'][0]}; {', '.join(sc['inv'])}. If worsening, go to {fac_en(c.loc, 'small')}.\n")
        d.t("-- ", dr_en(), P(hubdr, "init"), "\n")
        d.case = None
    return "telemedicine_consult", d


PKG = [("MG001A", "Acute febrile illness", "12,000"), ("MG064A", "Severe anaemia", "8,500"), ("MG006A", "Enteric fever", "15,000"),
       ("SG039A", "Cholecystectomy - laparoscopic", "38,500"), ("OR001B", "Hemiarthroplasty", "62,000"),
       ("MG027A", "Acute exacerbation of COPD", "18,000"), ("MG013A", "Pneumonia", "16,800"), ("MG046A", "Snake bite", "21,000")]


def en_pmjay():
    d = Doc(); c = Case("en"); p = c.pat
    sc = scen_en(c.g)
    dr = c.staff()
    med = c.staff(pool=c.pool)
    pk = ch(PKG)
    adm = c.ref
    addr, pin = addr_en(c.loc, ch(["in", "none", "sep"]))
    d.t(ch(["AB PM-JAY - PRE-AUTHORISATION REQUEST", "PMJAY Pre-auth note (TMS)", "Ayushman Bharat PM-JAY: pre-auth"]), "\n")
    d.t(f"Hospital: {fac_en(c.loc, 'big')}   Hospital ID: HOSP{ri(10, 36)}P{ri(100000, 999999)}\n")
    F = [(ch(["Beneficiary name", "Patient name", "Name"]), [hon_en(p), P(p, style=nst())]),
         (ch(["PM-JAY ID", "Beneficiary ID", "AB-PMJAY ID"]), [E(pmjay_id(), "MRN")]),
         ("Gender/Age", [agesex_en(c)]),
         (ch(["Aadhaar (e-KYC)", "Aadhaar", "Aadhaar no."]), [E(fmt_aadhaar(), "AADHAAR")]),
         (ch(L_MOB), [E(fmt_mobile(), "PHONE")]),
         (ch(L_ADDR), [E(addr, "ADDRESS")])]
    if opt(0.35):
        F.append((ch(["Alternate no.", "Res. phone", "Landline"]), [E(landline(), "PHONE")]))
    if pin and not addr.endswith(pin):
        F.append(("Pincode", [E(pin, "PINCODE")]))
    if opt(0.6):
        F.append(idproof_field(p))
    if opt(0.5):
        F.append((ch(["PAN (claim > Rs 50,000)", "PAN of card holder", "PAN"]), [E(pan_no(p), "PAN")]))
    if opt(0.45):
        F += abha_fields(c)
    F += [(ch(["IPD No.", "IP No."]), [E(ipd_no(), "MRN")]), ("Date of admission", [fdate(adm)]),
          ("Package", [f"{pk[0]} - {pk[1]}"]), ("Proposed amount", [f"Rs {pk[2]}"])]
    kv(d, F)
    d.t("\nClinical notes: ", P(p, "short") if opt(0.5) else "Patient", f" presented with {sc['cc']}. {sc['exam']}. ")
    d.t(f"Investigations: {'; '.join(sc['labs'][:2])}. Plan: {sc['rx'][0]}.\n")
    d.t("Documents attached: Aadhaar copy, admission notes, ", ch(["CBC report", "X-ray", "USG report"]), f". {neg_number()}\n")
    d.t("Treating doctor: ", dr_en(), P(dr), "   Aarogya Mitra: ", P(med, ch(["full", "short"])), "\n")
    if opt(0.5):
        d.t("Pre-auth no. (after approval): ", E(preauth_no(), "MRN"), "\n")
    if opt(0.5):
        d.t("Beneficiary e-mail for SMS/e-card: ", E(email_for(p), "EMAIL"), "\n")
    return "pmjay_preauth", d


# --------------------------------------------------------------------------
# HINDI TEMPLATES
# --------------------------------------------------------------------------
def hn(n):
    return str(n)


def scen_hi(g):
    S = [
        dict(dx="मलेरिया (पी. विवैक्स)", cc=f"{ri(3, 6)} दिन से ठंड लगकर बुखार", lab=[f"एमपी जांच: पी. विवैक्स पॉज़िटिव", f"हीमोग्लोबिन {hb()} ग्राम/डेसीलीटर"],
             rx=["टैबलेट क्लोरोक्वीन - तीन दिन की खुराक", "टैबलेट प्राइमाक्वीन 15 मि.ग्रा. रोज़ 14 दिन", "टैबलेट पैरासिटामोल 500 मि.ग्रा. बुखार होने पर"]),
        dict(dx="दस्त व उल्टी (गैस्ट्रोएंटेराइटिस)", cc="कल रात से पतले दस्त और उल्टी", lab=[f"सोडियम {ri(128, 136)} mEq/L", f"नाड़ी {ri(100, 120)}/मिनट"],
             rx=["ओआरएस घोल हर दस्त के बाद", "जिंक की गोली 20 मि.ग्रा. 14 दिन", "इंजेक्शन ओन्डेनसेट्रॉन 4 मि.ग्रा. ज़रूरत पर"]),
        dict(dx="फेफड़ों की टीबी (नया केस)", cc="दो हफ़्ते से ज़्यादा खांसी, शाम को बुखार, वज़न कम होना", lab=["बलगम सीबीनाट: एमटीबी पाया गया, रिफ़ प्रतिरोध नहीं", f"वज़न {ri(34, 50)} किलो"],
             rx=["4FDC (HRZE) 3 गोली रोज़ खाली पेट", "टैबलेट पाइरिडॉक्सिन 10 मि.ग्रा. रोज़"]),
        dict(dx="उच्च रक्तचाप", cc="सिर दर्द और चक्कर, बीपी की दवा बंद कर दी थी", lab=[f"बीपी {ri(160, 190)}/{ri(96, 112)} mmHg", f"क्रिएटिनिन {R.uniform(0.8, 1.5):.1f} mg/dL"],
             rx=["टैबलेट एम्लोडिपिन 5 मि.ग्रा. रोज़ सुबह", "टैबलेट टेल्मिसार्टन 40 मि.ग्रा. रोज़"]),
        dict(dx="मधुमेह (टाइप 2), अनियंत्रित", cc="बार-बार पेशाब, ज़्यादा प्यास, पैरों में जलन", lab=[f"शुगर (खाली पेट) {ri(150, 280)} mg/dL", f"HbA1c {R.uniform(8.0, 11.9):.1f} %"],
             rx=["टैबलेट मेटफॉर्मिन 500 मि.ग्रा. सुबह-शाम खाने के बाद", "टैबलेट ग्लिमेपिराइड 1 मि.ग्रा. नाश्ते से पहले"]),
        dict(dx="डेंगू बुखार", cc="तेज़ बुखार, आंखों के पीछे दर्द, बदन दर्द", lab=[f"प्लेटलेट {inr(ri(20, 95) * 1000)} /cmm", "एनएस-1 एंटीजन पॉज़िटिव"],
             rx=["टैबलेट पैरासिटामोल 500 मि.ग्रा. दिन में चार बार", "खूब पानी और ओआरएस", "दर्द निवारक (ब्रूफेन) न लें"]),
        dict(dx="सांप का काटना (न्यूरोटॉक्सिक)", cc="खेत में काम करते समय दाहिने पैर में सांप ने काटा, पलकें भारी", lab=["20 मिनट WBCT: खून जम गया", f"प्लेटलेट {plt()} /cmm"],
             rx=["एएसवी 10 वायल नस में", "इंजेक्शन नियोस्टिग्मिन + एट्रोपिन", "इंजेक्शन टीडी 0.5 मि.ली."]),
        dict(dx="निमोनिया", cc=f"{ri(4, 8)} दिन से बुखार, खांसी और सांस फूलना", lab=[f"टीएलसी {inr(ri(120, 210) * 100)} /cmm", f"ऑक्सीजन {ri(88, 94)}%"],
             rx=["इंजेक्शन सेफ्ट्रियाक्सोन 1 ग्राम दिन में दो बार", "टैबलेट एज़िथ्रोमाइसिन 500 मि.ग्रा. 5 दिन"]),
    ]
    if g == "F":
        S.append(dict(dx="गर्भावस्था में गंभीर खून की कमी", cc="कमज़ोरी, सांस फूलना, पैरों में सूजन",
                      lab=[f"हीमोग्लोबिन {R.uniform(5.2, 6.9):.1f} ग्राम/डेसीलीटर", f"बीपी {ri(100, 120)}/{ri(64, 80)}"],
                      rx=["आयरन सुक्रोज़ इंजेक्शन 200 मि.ग्रा. (तीन खुराक)", "आयरन-फोलिक एसिड की गोली दिन में दो बार", "कैल्शियम की गोली"]))
    return ch(S)


def hi_age(c):
    sex = "पुरुष" if c.g == "M" else "महिला"
    return ch([f"{c.age} वर्ष / {sex}", f"{c.age} वर्ष, {sex}", f"{c.age}/{'पु' if c.g == 'M' else 'म'}", f"{c.age} साल"])


def hi_guardian(c):
    kind, rel = c.guardian()
    if kind == "husband":
        lab = ch(["पति का नाम", "पति", "पत्नी", "पति/पिता का नाम"])
        if lab == "पत्नी":
            lab = "पति का नाम"
    else:
        lab = ch(["पिता का नाम", "पिता", "पिता/पति का नाम"])
    return (lab, [hon_hi(rel, prob=0.35), P(rel)])


def hi_guardian_inline(c):
    kind, rel = c.guardian()
    w = ch(["पत्नी", "धर्मपत्नी", "w/o"]) if kind == "husband" else (ch(["पुत्र", "s/o"]) if c.g == "M" else ch(["पुत्री", "d/o"]))
    return [w, " ", hon_hi(rel, prob=0.4), P(rel)]


def hi_mob(dev):
    return E(fmt_mobile(ch(["plain", "plain", "split", "p91s", "zero", "dash"]), dev=dev and opt(0.8)), "PHONE")


def hi_aadhaar(dev):
    return E(fmt_aadhaar(dev=dev and opt(0.8)), "AADHAAR")


HL_NAME = ["नाम", "मरीज़ का नाम", "रोगी का नाम", "नाम"]
HL_ADDR = ["पता", "पता", "निवास", "स्थायी पता"]
HL_MOB = ["मोबाइल", "मोबाइल नं.", "मो.", "फ़ोन", "संपर्क नं."]
HL_AADH = ["आधार", "आधार नं.", "आधार संख्या", "आधार कार्ड नं."]
HL_DOB = ["जन्म तिथि", "जन्मतिथि", "जन्म दिनांक"]


def hdate(dt, dev):
    s = fdate(dt, ch(["dmy/", "dmy-", "dmy.", "hi"]))
    return s


def hi_doc():
    d = Doc()
    d.deva = opt(0.35)
    return d


def hi_dr():
    return ch(["डॉ. ", "डॉ ", "डॉ० "])


def hi_opd():
    d = hi_doc(); c = Case("hi"); p = c.pat
    sc = scen_hi(c.g); dr = c.staff()
    addr, pin = addr_hi(c.loc, ch(["in", "none", "sep"]))
    d.t(fac_hi(c.loc), "\n", ch(["ओपीडी पर्ची", "बाह्य रोगी पर्ची", "ओ.पी.डी. रजिस्ट्रेशन पर्ची"]), "\n")
    F = [(ch(["ओपीडी नं.", "पंजीकरण सं.", "रजि. नं."]), [E(opd_no(), "MRN")]), ("दिनांक", [hdate(c.ref, d.deva)]),
         (ch(HL_NAME), [hon_hi(p), P(p)]), (ch(["आयु/लिंग", "उम्र", "आयु"]), [hi_age(c)]), hi_guardian(c),
         (ch(HL_ADDR), [E(addr, "ADDRESS")])]
    if pin and not addr.endswith(pin) and opt(0.6):
        F.append((ch(["पिन", "पिन कोड"]), [E(pin, "PINCODE")]))
    F.append((ch(HL_MOB), [hi_mob(d.deva)]))
    if opt(0.35):
        F.append((ch(HL_AADH), [hi_aadhaar(d.deva)]))
    if opt(0.25):
        F.append((ch(["आभा नं.", "आभा नंबर", "ABHA"]), [E(abha_number(), "ABHA_NUMBER")]))
        if opt(0.6):
            F.append(("आभा पता", [E(abha_address(p, c.dob.year), "ABHA_ADDRESS")]))
    kv(d, F, pair=0.25)
    d.t("\n", ch(["शिकायत: ", "तकलीफ़: ", "मुख्य शिकायत - "]), sc["cc"], "\n")
    d.t(ch(["जांच: ", "परीक्षण: "]), "; ".join(sc["lab"]), "\n")
    d.t(ch(["दवाइयाँ:", "उपचार:", "Rx"]), "\n")
    for i, r in enumerate(sc["rx"], 1):
        d.t(f"{i}. {r}\n")
    d.t(ch([f"{ri(3, 7)} दिन बाद दोबारा दिखाएं।", "तकलीफ़ बढ़ने पर तुरंत आएं।", f"अगली जांच {hdate(c.ref + timedelta(days=ri(5, 14)), d.deva)} को।"]), "\n")
    if opt(0.5):
        pre_rel = ch(["बेटा", "बेटी", "भाई", "बहू"])
        a = relative(p, "M" if pre_rel in ("बेटा", "भाई") else "F")
        d.t(f"मरीज़ के साथ {pre_rel} ", P(a, ch(["full", "short"])), " आए थे।\n")
    d.t(hi_dr(), P(dr), ch([" (चिकित्सा अधिकारी)", ", एमबीबीएस", ""]), "\n")
    return "opd_prescription", d


def hi_discharge():
    d = hi_doc(); c = Case("hi"); p = c.pat
    sc = scen_hi(c.g); dr1, dr2 = c.staff(), c.staff()
    addr, pin = addr_hi(c.loc, ch(["in", "in", "none"]))
    adm = c.ref - timedelta(days=ri(2, 10))
    d.t(fac_hi(c.loc, "big"), "\n", ch(["डिस्चार्ज सारांश", "छुट्टी पर्ची", "डिस्चार्ज टिकट"]), "\n\n")
    F = [(ch(["आईपीडी नं.", "भर्ती क्रमांक", "भर्ती सं."]), [E(ipd_no(), "MRN")]), (ch(["यूएचआईडी", "UHID"]), [E(uhid(), "MRN")]),
         (ch(HL_NAME), [hon_hi(p), P(p)]), ("आयु/लिंग", [hi_age(c)]), hi_guardian(c), (ch(HL_ADDR), [E(addr, "ADDRESS")]),
         (ch(HL_MOB), [hi_mob(d.deva)])]
    if opt(0.4):
        F.append((ch(HL_DOB), [E(hdate(c.dob, d.deva), "DOB")]))
    if opt(0.45):
        F.append((ch(HL_AADH), [hi_aadhaar(d.deva)]))
    if opt(0.3):
        F.append((ch(["आभा नंबर", "आभा आईडी"]), [E(abha_number(), "ABHA_NUMBER")]))
        if opt(0.6):
            F.append(("आभा पता", [E(abha_address(p, c.dob.year), "ABHA_ADDRESS")]))
    if opt(0.2):
        F.append(("पैन", [E(pan_no(p), "PAN")]))
    if opt(0.2):
        F.append(("ईमेल", [E(email_for(p), "EMAIL")]))
    F += [("बेड नं.", [f"{ri(1, 40)}, {ch(['मेडिसिन वार्ड', 'महिला वार्ड', 'सर्जिकल वार्ड', 'आईसीयू'])}"]),
          ("भर्ती दिनांक", [hdate(adm, d.deva)]), ("छुट्टी दिनांक", [hdate(c.ref, d.deva)]),
          (ch(["चिकित्सक", "इलाज करने वाले डॉक्टर"]), [hi_dr(), P(dr1)])]
    kv(d, F)
    d.t("\nनिदान: ", sc["dx"], "\n")
    rel_w = ch(["बेटा", "बेटी", "पति" if c.g == "F" else "पत्नी", "भाई", "भतीजा"])
    a = relative(p, "F" if rel_w in ("बेटी", "पत्नी") else "M")
    d.t("विवरण: ", P(p, "short"), f" को {sc['cc']} की शिकायत के साथ उनके {rel_w} ", P(a, ch(["short", "full"])), " भर्ती कराने लाए। ")
    d.t("जांच: ", "; ".join(sc["lab"]), "। ")
    d.t("इलाज के बाद हालत में सुधार है, ", P(p, "short") if opt(0.5) else "मरीज़", " को छुट्टी दी जा रही है।\n\n")
    d.t("घर पर दवाइयाँ:\n")
    for i, r in enumerate(sc["rx"], 1):
        d.t(f"{i}. {r}\n")
    d.t(f"\n{hdate(c.ref + timedelta(days=ri(5, 14)), d.deva)} को ओपीडी में दिखाएं। आपात स्थिति में 108 पर कॉल करें।\n")
    d.t("तैयारकर्ता: ", hi_dr(), P(dr2), "    हस्ताक्षर: ", hi_dr(), P(dr1, "last"), "\n")
    return "discharge_summary", d


def hi_referral():
    d = hi_doc(); c = Case("hi"); p = c.pat
    sc = scen_hi(c.g); mo = c.staff()
    src, dst = fac_hi(c.loc, "small"), fac_hi(c.loc, "big")
    addr, pin = addr_hi(c.loc, ch(["in", "sep"]))
    d.t(ch(["रेफरल पर्ची", "रेफर पत्र", "उच्च संस्थान हेतु रेफरल"]), f"\nदिनांक: {hdate(c.ref, d.deva)}\n\n")
    d.t(f"सेवा में,\nचिकित्सा अधीक्षक,\n{dst}\n\n")
    d.t("महोदय,\nमरीज़ ", hon_hi(p), P(p), f", आयु {c.age} वर्ष, ", hi_guardian_inline(c), ", निवासी ", E(addr, "ADDRESS"))
    if pin and not addr.endswith(pin):
        d.t(" (पिन ", E(pin, "PINCODE"), ")")
    d.t(f", दिनांक {hdate(c.ref, d.deva)} को {src} में {sc['cc']} की शिकायत लेकर आए। ")
    d.t(f"प्रारंभिक निदान: {sc['dx']}। यहां दी गई दवा: {sc['rx'][0]}। ")
    d.t("आगे के इलाज हेतु आपके संस्थान में रेफर किया जा रहा है।\n")
    rel_w = ch(["बेटा", "पति" if c.g == "F" else "पत्नी", "भाई", "देवर", "बहू"])
    a = relative(p, "F" if rel_w in ("पत्नी", "बहू") else "M")
    d.t(f"साथ में {rel_w} ", P(a, ch(["short", "full"])), " (मो. ", hi_mob(d.deva), ")। ")
    if opt(0.5):
        d.t("आधार: ", hi_aadhaar(d.deva), "। ")
    d.t(ch(["108 एम्बुलेंस से भेजा गया।", "102 वाहन से भेजा गया।"]), "\n\n")
    d.t(hi_dr(), P(mo), f"\nचिकित्सा अधिकारी, {src}\n")
    if opt(0.4):
        d.t("सूचना: आशा ", P(c.staff("F", pool="hi_hindu"), "short"), " को फ़ॉलो-अप हेतु।\n")
    return "referral_letter", d


def hi_asha():
    d = hi_doc(); c = Case("hi", g="F", age=(19, 34)); p = c.pat
    hus = relative(p, "M"); c._rel = ("husband", hus)
    asha, anm = c.staff("F", pool="hi_hindu"), c.staff("F")
    addr, pin = addr_hi(c.loc, ch(["in", "none", "sep"]))
    lay = ch(["hbnc", "prose"])
    if lay == "hbnc":
        d.t(ch(["आशा गृह भ्रमण रिपोर्ट (एचबीएनसी)", "गृह आधारित नवजात देखभाल - भ्रमण प्रपत्र"]), "\n")
        bdob = c.ref - timedelta(days=ri(3, 21))
        F = [("आशा का नाम", [P(asha)]), ("उपकेंद्र", [c.loc.po]), ("भ्रमण दिनांक", [hdate(c.ref, d.deva)]),
             ("माता का नाम", [hon_hi(p, prob=0.4), P(p)]), ("पति का नाम", [P(hus)]), (ch(HL_ADDR), [E(addr, "ADDRESS")])]
        if pin and not addr.endswith(pin):
            F.append(("पिन कोड", [E(pin, "PINCODE")]))
        F += [(ch(HL_MOB), [hi_mob(d.deva)]), ("शिशु", [ch(["बेबी ऑफ़ ", "शिशु - "]), P(p, "short"), "  ", ch(["लड़का", "लड़की"])]),
              ("शिशु की जन्म तिथि", [E(hdate(bdob, d.deva), "DOB")]), ("जन्म के समय वज़न", [f"{R.uniform(1.9, 3.4):.1f} किलो"])]
        if opt(0.5):
            F.append(("आरसीएच आईडी", [E(rch_id(), "MRN")]))
        kv(d, F)
        d.t("\nनिष्कर्ष: ", ch(["शिशु ठीक से दूध पी रहा है, नाल सूखी है", "शिशु को हल्का पीलिया, पीएचसी रेफर", "शिशु का तापमान सामान्य, वज़न बढ़ा"]), "। ")
        d.t("माँ को केवल स्तनपान और कंगारू मदर केयर की सलाह दी। प्रोत्साहन राशि ₹250 का दावा।\n")
        d.t("एएनएम: ", P(anm), "\n")
    else:
        d.t(f"आशा रिपोर्ट, दिनांक {hdate(c.ref, d.deva)}\n")
        d.t("आज ", P(p), " पत्नी ", P(hus), ", ", E(addr, "ADDRESS"), " के घर भ्रमण किया। ")
        d.t(f"वह {ri(6, 9)} महीने की गर्भवती हैं। पैरों में सूजन और सिर दर्द की शिकायत, बीपी {ri(140, 160)}/{ri(90, 104)}। ")
        d.t("उनका भाई ", P(relative(p, "M"), "short"), " उन्हें पिछले महीने वीएचएसएनडी सत्र में लाया था। ")
        d.t("पति का मोबाइल ", hi_mob(d.deva), "। ")
        if opt(0.5):
            d.t("आधार ", hi_aadhaar(d.deva), "। ")
        d.t(f"तुरंत {fac_hi(c.loc, 'small')} जाने की सलाह दी। हीमोग्लोबिन {hb()} ग्राम।\n")
        d.t("आशा: ", P(asha), "\n")
    return "asha_home_visit", d


def hi_anc():
    d = hi_doc(); c = Case("hi", g="F", age=(18, 36)); p = c.pat
    hus = relative(p, "M"); c._rel = ("husband", hus)
    asha, anm = c.staff("F", pool="hi_hindu"), c.staff("F")
    lmp = c.ref - timedelta(days=ri(60, 250)); edd = lmp + timedelta(days=280)
    addr, pin = addr_hi(c.loc, ch(["in", "sep", "none"]))
    d.t(ch(["मातृ एवं शिशु सुरक्षा कार्ड", "प्रसव पूर्व जांच (एएनसी) कार्ड", "एएनसी रजिस्टर"]), "\n")
    d.t(f"केंद्र: {fac_hi(c.loc, 'small')}\n")
    F = [("आरसीएच आईडी", [E(rch_id(), "MRN")]), (ch(["गर्भवती महिला का नाम", "नाम"]), [hon_hi(p, prob=0.4), P(p)]),
         ("आयु", [f"{c.age} वर्ष"]), ("पति का नाम", [P(hus)]), (ch(HL_ADDR), [E(addr, "ADDRESS")])]
    if pin and not addr.endswith(pin):
        F.append(("पिन", [E(pin, "PINCODE")]))
    F.append((ch(HL_MOB), [hi_mob(d.deva)]))
    if opt(0.55):
        F.append((ch(HL_AADH), [hi_aadhaar(d.deva)]))
    if opt(0.3):
        F.append((ch(HL_DOB), [E(hdate(c.dob, d.deva), "DOB")]))
    if opt(0.35):
        F.append(("आभा पता", [E(abha_address(p, c.dob.year), "ABHA_ADDRESS")]))
    if opt(0.2):
        F.append(("मतदाता पहचान पत्र", [E(epic_no(), "VOTER_ID")]))
    F += [("अंतिम माहवारी (LMP)", [hdate(lmp, d.deva)]), ("संभावित प्रसव तिथि (EDD)", [hdate(edd, d.deva)]),
          ("गर्भ", [f"G{ri(1, 4)} P{ri(0, 3)}"]), ("रक्त समूह", [ch(["B+", "O+", "A+", "AB+"])])]
    kv(d, F)
    d.t("\nजांच विवरण:\n")
    dt = lmp + timedelta(days=ri(60, 90))
    for _ in range(ri(2, 3)):
        if dt > c.ref:
            break
        d.t(f"{fdate(dt, 'dmy/')} - वज़न {ri(42, 66)} किलो, बीपी {ri(100, 150)}/{ri(64, 96)}, Hb {hb()}, FHR {ri(128, 154)}\n")
        dt += timedelta(days=ri(28, 45))
    d.t(f"टीडी-1 दिया ({neg_number()}), आयरन की {ch([90, 180])} गोलियाँ और कैल्शियम दिया।\n")
    d.t("उच्च जोखिम: ", ch(["नहीं", "गंभीर एनीमिया", "पिछला ऑपरेशन से प्रसव", "उच्च रक्तचाप"]), "\n")
    d.t("आशा: ", P(asha), " (", hi_mob(d.deva), ")   एएनएम: ", P(anm, ch(["full", "short"])), "\n")
    return "anc_card", d


def hi_tb():
    d = hi_doc(); c = Case("hi"); p = c.pat
    sup, mo = c.staff(pool="hi_hindu"), c.staff()
    fam = relative(p, "F" if c.g == "M" else "M")
    start = c.ref - timedelta(days=ri(56, 170))
    addr, pin = addr_hi(c.loc, ch(["in", "sep"]))
    d.t(ch(["टीबी उपचार फ़ॉलो-अप (निक्षय)", "राष्ट्रीय टीबी उन्मूलन कार्यक्रम - फ़ॉलो-अप कार्ड"]), "\n")
    F = [("निक्षय आईडी", [E(nikshay_id(), "MRN")]), (ch(HL_NAME), [hon_hi(p), P(p)]), ("आयु/लिंग", [hi_age(c)]),
         (ch(HL_ADDR), [E(addr, "ADDRESS")])]
    if pin and not addr.endswith(pin):
        F.append(("पिन कोड", [E(pin, "PINCODE")]))
    F += [(ch(HL_MOB), [hi_mob(d.deva)])]
    if opt(0.4):
        F.append((ch(HL_AADH), [hi_aadhaar(d.deva)]))
    elif opt(0.4):
        F.append(("वोटर आईडी", [E(epic_no(), "VOTER_ID")]))
    if opt(0.25):
        F.append(("आभा नं.", [E(abha_number(), "ABHA_NUMBER")]))
    F += [("उपचार शुरू", [hdate(start, d.deva)]), ("उपचार सहयोगी", [P(sup), ch([" (आशा)", " (आंगनबाड़ी कार्यकर्ता)", ""])])]
    kv(d, F)
    w0 = ri(34, 50)
    d.t(f"\nफ़ॉलो-अप {hdate(c.ref, d.deva)}: वज़न {w0} से बढ़कर {w0 + ri(1, 5)} किलो, बलगम जांच {ch(['नेगेटिव', 'नेगेटिव', 'पॉज़िटिव (स्कैंटी)'])}। ")
    d.t(f"दवा नियमित ({ri(88, 100)}%)। निक्षय पोषण योजना की ₹1000 राशि खाते में आई। ")
    d.t(ch(["पत्नी ", "घर वाली "]) if c.g == "M" else "पति ", P(fam, "short"), " की जांच की गई, लक्षण नहीं।\n")
    d.t("अगली मुलाक़ात: ", hdate(c.ref + timedelta(days=30), d.deva), "। ", hi_dr(), P(mo), "\n")
    return "tb_nikshay_followup", d


def hi_imm():
    d = hi_doc(); c = Case("hi", age=0); child = c.pat
    mother = make_person(c.pool, "F", family=child.family, hindi=True)
    father = relative(child, "M")
    anm = c.staff("F")
    bdob = c.ref - timedelta(days=ri(45, 300))
    addr, pin = addr_hi(c.loc, ch(["in", "none"]))
    d.t(ch(["टीकाकरण कार्ड", "नियमित टीकाकरण रिकॉर्ड", "बच्चे का टीकाकरण विवरण"]), "\n")
    F = [("आरसीएच आईडी", [E(rch_id(), "MRN")])]
    if opt(0.6):
        F.append(("बच्चे का नाम", [P(child, ch(["short", "full"]))]))
    else:
        F.append(("बच्चे का नाम", ["बेबी ऑफ़ ", P(mother)]))
    F += [("जन्म तिथि", [E(hdate(bdob, d.deva), "DOB")]), ("लिंग", ["लड़का" if c.g == "M" else "लड़की"]), ("माता का नाम", [P(mother)]),
          ("पिता का नाम", [hon_hi(father, prob=0.4), P(father)]), (ch(HL_ADDR), [E(addr, "ADDRESS")]), (ch(HL_MOB), [hi_mob(d.deva)])]
    kv(d, F)
    d.t("\nटीका | दिनांक | बैच\n")
    age_days = (c.ref - bdob).days
    for v, due in VACC:
        off = {"at birth": 0, "6 wk": 42, "10 wk": 70, "14 wk": 98, "9 mo": 270}[due]
        if off > age_days:
            break
        d.t(f"{v} | {fdate(bdob + timedelta(days=off + ri(0, 9)), 'dmy/')} | {ch('ABCDKLM')}{ri(2300, 2699)}{ri(100, 999)}\n")
    d.t(f"\nटीके के बाद {ch(['कोई दिक्कत नहीं', 'हल्का बुखार, ठीक हो गया'])}। टीकाकर्ता एएनएम: ", P(anm), "\n")
    return "immunisation_record", d


def hi_birth():
    d = hi_doc(); c = Case("hi", g="F", age=(19, 36)); m = c.pat
    f = relative(m, "M"); sn = c.staff("F")
    addr, pin = addr_hi(c.loc, ch(["in", "sep"]))
    d.t(ch(["जन्म सूचना (प्रारूप) - सीआरएस हेतु", "जन्म रिपोर्ट प्रपत्र-1 (मसौदा)"]), "\n")
    d.t(f"जन्म स्थान: प्रसव कक्ष, {fac_hi(c.loc)}\n")
    F = [("जन्म तिथि", [E(hdate(c.ref, d.deva), "DOB")]), ("समय", [f"{ri(1, 12)}:{ri(0, 59):02d} {ch(['प्रातः', 'सायं', 'रात्रि'])}"]),
         ("लिंग", [ch(["बालक", "बालिका"])]), ("वज़न", [f"{R.uniform(1.9, 3.8):.1f} किलो"]),
         ("माता का नाम", [hon_hi(m, prob=0.4), P(m)]), ("माता की आयु", [f"{c.age} वर्ष"]), ("पिता का नाम", [P(f)])]
    if opt(0.6):
        F.append(("माता का आधार", [hi_aadhaar(d.deva)]))
    if opt(0.3):
        F.append(("पिता का मतदाता पहचान पत्र", [E(epic_no(), "VOTER_ID")]))
    F.append(("माता-पिता का पता", [E(addr, "ADDRESS")]))
    if pin and not addr.endswith(pin):
        F.append(("पिन", [E(pin, "PINCODE")]))
    F.append((ch(HL_MOB), [hi_mob(d.deva)]))
    kv(d, F)
    d.t("प्रसव कराने वाली: स्टाफ़ नर्स ", P(sn), f"। जेएसवाई ₹{ch(['1400', '700'])} देय। बीसीजी, ओपीवी-0, हेप-बी दिए।\n")
    return "birth_notification_draft", d


def hi_death():
    d = hi_doc(); c = Case("hi", age=(45, 92)); p = c.pat
    inf = relative(p, "M"); drc = c.staff()
    addr, pin = addr_hi(c.loc, ch(["in", "none"]))
    sc = scen_hi(c.g)
    d.t(ch(["मृत्यु सूचना (प्रारूप)", "मृत्यु के कारण का चिकित्सा प्रमाण पत्र - मसौदा"]), "\n")
    d.t(f"अस्पताल: {fac_hi(c.loc, 'big')}, वार्ड/बेड {ri(1, 30)}\n")
    F = [("मृतक का नाम", [ch(["स्व. ", "स्वर्गीय ", ""]), P(p)]), ("आयु", [f"{c.age} वर्ष"]), ("लिंग", ["पुरुष" if c.g == "M" else "महिला"])]
    if opt(0.5):
        F.append((ch(HL_DOB), [E(hdate(c.dob, d.deva), "DOB")]))
    F += [hi_guardian(c), ("स्थायी पता", [E(addr, "ADDRESS")]),
          ("मृत्यु की तिथि व समय", [f"{hdate(c.ref, d.deva)}, {ri(1, 12)}:{ri(0, 59):02d}"]), ("भर्ती क्रमांक", [E(ipd_no(), "MRN")])]
    kv(d, F)
    d.t(f"\nमृत्यु का कारण: (क) तत्काल: {ch(['सेप्टिक शॉक', 'सांस की विफलता', 'हृदय गति रुकना'])} (ख) पूर्ववर्ती: {sc['dx']}\n")
    d.t("सूचनादाता: ", P(inf), f" ({ch(['पुत्र', 'भाई', 'भतीजा', 'पौत्र'])}), मो. ", hi_mob(d.deva))
    if opt(0.5):
        d.t(", मतदाता पहचान पत्र ", E(epic_no(), "VOTER_ID"))
    elif opt(0.5):
        d.t(", आधार ", hi_aadhaar(d.deva))
    d.t("\nप्रमाणित करने वाले चिकित्सक: ", hi_dr(), P(drc), "\n")
    return "death_notification_draft", d


def hi_pmjay():
    d = hi_doc(); c = Case("hi"); p = c.pat
    sc = scen_hi(c.g); dr = c.staff(); mitra = c.staff(pool="hi_hindu")
    addr, pin = addr_hi(c.loc, ch(["in", "sep"]))
    pk = ch(PKG)
    d.t(ch(["आयुष्मान भारत पीएम-जय - प्री-ऑथ अनुरोध", "आयुष्मान कार्ड - भर्ती पूर्व अनुमोदन नोट"]), "\n")
    d.t(f"अस्पताल: {fac_hi(c.loc, 'big')}\n")
    F = [("लाभार्थी का नाम", [hon_hi(p), P(p)]), ("पीएम-जय आईडी", [E(pmjay_id(), "MRN")]), ("आयु/लिंग", [hi_age(c)]),
         (ch(HL_AADH), [hi_aadhaar(d.deva)]), (ch(HL_MOB), [hi_mob(d.deva)]), (ch(HL_ADDR), [E(addr, "ADDRESS")])]
    if pin and not addr.endswith(pin):
        F.append(("पिन कोड", [E(pin, "PINCODE")]))
    if opt(0.5):
        F.append(("मतदाता पहचान पत्र", [E(epic_no(), "VOTER_ID")]))
    if opt(0.3):
        F.append(("पैन नं.", [E(pan_no(p), "PAN")]))
    if opt(0.25):
        F.append(("ई-मेल", [E(email_for(p), "EMAIL")]))
    if opt(0.3):
        F.append(("आभा नंबर", [E(abha_number(), "ABHA_NUMBER")]))
    F += [("भर्ती क्रमांक", [E(ipd_no(), "MRN")]), ("पैकेज", [Raw(f"{pk[0]} - {pk[1]}")]), ("अनुमानित राशि", [f"₹{pk[2]}"])]
    kv(d, F)
    d.t(f"\nनैदानिक विवरण: {sc['cc']}। निदान: {sc['dx']}।\n")
    d.t("इलाज करने वाले डॉक्टर: ", hi_dr(), P(dr), "   आरोग्य मित्र: ", P(mitra, ch(["full", "short"])), "\n")
    return "pmjay_preauth", d


# --------------------------------------------------------------------------
# MIXED (Hinglish / code-switched) TEMPLATES
# --------------------------------------------------------------------------
def scen_mx(g):
    S = [
        dict(dx="Malaria (P. vivax)", cc=f"{ri(3, 6)} din se thand lagke bukhar aa raha hai", hi="ठंड लगकर बुखार",
             rx=["Tab Chloroquine as per schedule", "Tab Primaquine 15 mg roz 14 din", "Tab PCM 650 bukhar hone par"], lab=f"MP RDT positive, Hb {hb()}"),
        dict(dx="Acute gastroenteritis", cc="kal raat se loose motion aur ulti, 8-10 baar", hi="दस्त और उल्टी",
             rx=["ORS har loose motion ke baad", "Zinc 20 mg 14 din", "Tab Ondansetron 4 mg SOS"], lab=f"Pulse {ri(100, 120)}/min, BP {ri(90, 110)}/{ri(60, 70)}"),
        dict(dx="Pulmonary Koch's", cc="2 hafte se zyada khansi, raat ko pasina, vajan kam ho raha hai", hi="खांसी और वज़न कम",
             rx=["4FDC 3 tab roz khali pet", "Tab Pyridoxine 10 mg"], lab="CBNAAT MTB detected, Rif sensitive"),
        dict(dx="Hypertension, uncontrolled", cc="sar dard aur chakkar, BP ki dawai band kar di thi", hi="सिर दर्द और चक्कर",
             rx=["Tab Amlodipine 5 mg subah", "Tab Telmisartan 40 mg"], lab=f"BP {ri(160, 190)}/{ri(96, 112)} mmHg"),
        dict(dx="T2DM uncontrolled", cc="baar baar peshab, pyaas zyada, pair mein jalan", hi="बार-बार पेशाब",
             rx=["Tab Metformin 500 mg BD khane ke baad", "Tab Glimepiride 1 mg"], lab=f"FBS {ri(160, 280)} mg/dL"),
        dict(dx="Dengue (NS1 positive)", cc="tez bukhar, aankhon ke peeche dard, badan toot raha hai", hi="तेज़ बुखार",
             rx=["Tab PCM 500 QID", "pani aur ORS khoob", "Brufen/Combiflam bilkul nahi"], lab=f"Platelet {inr(ri(20, 95) * 1000)}"),
        dict(dx="URTI with fever", cc="gale mein dard, naak beh rahi hai, halka bukhar", hi="गले में दर्द",
             rx=["Tab Cetirizine 10 mg raat ko", "garam pani ke gargle", "Tab PCM SOS"], lab=f"Temp {R.uniform(99.2, 101.5):.1f} F"),
    ]
    if g == "F":
        S.append(dict(dx="Anaemia in pregnancy", cc="kamzori, saans phoolna, pairon mein soojan", hi="पैरों में सूजन",
                      rx=["IFA tab roz do baar", "Calcium tab", "Inj Iron sucrose 200 mg"], lab=f"Hb {R.uniform(6.0, 8.5):.1f} g/dL"))
    return ch(S)


def mx_cc(sc):
    """complaint in Latin Hinglish, or English with a Devanagari fragment."""
    if opt(0.35):
        return f"patient complains of {sc['hi']} since {ri(2, 6)} days"
    return sc["cc"]


ML_NAME = ["Naam", "Name", "Mareez ka naam", "Patient"]
ML_ADDR = ["Pata", "Address", "Ghar ka pata", "Add"]
ML_MOB = ["Mob", "Mobile", "Phone no", "Number"]


def mx_guardian(c):
    kind, rel = c.guardian()
    lab = ch(["Pati ka naam", "Husband", "W/o"]) if kind == "husband" else ch(["Pita ka naam", "Father", "S/o" if c.g == "M" else "D/o"])
    return (lab, [hon_en(rel, prob=0.2), P(rel)])


def mx_tele():
    d = Doc(); c = Case("mixed"); p = c.pat
    sc = scen_mx(c.g); dr, cho = c.staff(), c.staff(pool=c.pool)
    d.case = ch([None, None, "lower"])
    d.t(ch(["eSanjeevani consult", "Teleconsult note", "Tele-consultation (HWC)"]), " | ID ", E(esj_id(), "MRN"), "\n")
    d.t(f"Spoke: HWC {c.loc.po}, CHO ", P(cho, ch(["full", "short"])), "\n")
    d.t("Patient: ", P(p), f", {c.age}/{c.g}, mob ", E(fmt_mobile(), "PHONE"), "\n")
    if opt(0.7):
        d.t("ABHA: ", E(abha_number(), "ABHA_NUMBER"))
        if opt(0.7):
            d.t("  (", E(abha_address(p, c.dob.year), "ABHA_ADDRESS"), ")")
        d.t("\n")
    if opt(0.35):
        d.t(ch(["DOB ", "Janm tithi "]), E(fdate(c.dob), "DOB"), ch(["\n", ", "]))
    if opt(0.25):
        d.t("email ", E(email_for(p), "EMAIL"), "\n")
    rk = ch(["wife", "beti", "beta", "bahu", "bhai"]) if c.g == "M" else ch(["pati", "beta", "saas", "bhai", "beti"])
    a = relative(p, "F" if rk in ("wife", "beti", "bahu", "saas") else "M")
    d.t("CC: ", mx_cc(sc), ". ", ch([f"{rk.capitalize()} ", f"Saath mein {rk} "]), P(a, ch(["short", "full"])),
        ch([" ne bataya ki dawai time pe nahi le rahe the.", " bhi call pe the.", " ko samjhaya gaya."]), "\n")
    d.t(f"O/E (by CHO): {sc['lab']}\n")
    d.t("Adv:\n")
    for r in sc["rx"]:
        d.t("- ", r, "\n")
    d.t(ch([f"{ri(3, 5)} din mein aaram na ho to PHC aayein.", "Agar saans phoole to turant 108 call karein.", "दवाई समय पर लें, follow up after 1 week."]), "\n")
    d.t("Hub doctor: ", dr_en(), P(dr, ch(["full", "init"])), "\n")
    return "telemedicine_consult", d


def mx_asha():
    d = Doc(); c = Case("mixed", g="F", age=(19, 34)); p = c.pat
    hus = relative(p, "M"); c._rel = ("husband", hus)
    asha = c.staff("F", pool=c.pool)
    addr, pin = addr_en(c.loc, ch(["in", "none", "sep"]), hinglish=True)
    d.case = ch([None, "lower"])
    d.t(f"ASHA visit report - {fdate(c.ref)}\n")
    d.t("ASHA ", P(asha), " ne aaj ", P(p), " (pati ", P(hus, ch(["full", "short"])), ") ke ghar visit kiya. ")
    d.t("Pata: ", E(addr, "ADDRESS"), ". ")
    if pin and not addr.endswith(pin):
        d.t("pincode ", E(pin, "PINCODE"), ". ")
    d.t(f"Woh {ri(5, 9)} mahine pregnant hai, {ch(['पैरों में सूजन', 'सिर दर्द', 'kamzori'])} ki shikayat. BP {ri(120, 160)}/{ri(80, 104)}, Hb {hb()}. ")
    d.t("Uski saas ", P(make_person(c.pool, "F", family=hus.family), "short"), " bhi saath thi. ")
    d.t("Pati ka number ", E(fmt_mobile(), "PHONE"), ". ")
    if opt(0.5):
        d.t("Aadhar ", E(fmt_aadhaar(), "AADHAAR"), ". ")
    d.t(f"RCH ID {'' if opt(0.5) else 'no. '}", E(rch_id(), "MRN"), f". {fac_en(c.loc, 'small')} jaane ki salah di, JSY form bhara.\n")
    return "asha_home_visit", d


def mx_opd():
    d = Doc(); c = Case("mixed"); p = c.pat
    sc = scen_mx(c.g); dr = c.staff()
    addr, pin = addr_en(c.loc, ch(["in", "none", "sep"]), short=opt(0.5), hinglish=True)
    d.t(fac_en(c.loc, "small"), " - OPD parchi\n")
    F = [("OPD no", [E(opd_no(), "MRN")]), ("Tarikh", [fdate(c.ref)]), (ch(ML_NAME), [hon_en(p, prob=0.4), P(p, style=nst())]),
         ("Umar", [f"{c.age} saal, {'purush' if c.g == 'M' else 'mahila'}"]), mx_guardian(c), (ch(ML_ADDR), [E(addr, "ADDRESS")])]
    if pin and not addr.endswith(pin):
        F.append(("Pin", [E(pin, "PINCODE")]))
    F.append((ch(ML_MOB), [E(fmt_mobile(), "PHONE")]))
    if opt(0.3):
        F.append(("Aadhar no", [E(fmt_aadhaar(), "AADHAAR")]))
    if opt(0.35):
        F.append((ch(["Voter card", "Voter ID", "EPIC"]), [E(epic_no(), "VOTER_ID")]))
    if opt(0.35):
        F.append((ch(["Janm tithi", "DOB"]), [E(fdate(c.dob), "DOB")]))
    if opt(0.2):
        F.append(("Email", [E(email_for(p), "EMAIL")]))
    kv(d, F)
    d.t("\nShikayat: ", mx_cc(sc), "\nJaanch: ", sc["lab"], "\nDawai:\n")
    for i, r in enumerate(sc["rx"], 1):
        d.t(f"{i}. {r}\n")
    d.t(ch(["3 din baad dikhaye", "Follow up after 5 days", "Aaram, khoob pani piyein"]), ". ", dr_en(), P(dr, "last"), "\n")
    return "opd_prescription", d


def mx_discharge():
    d = Doc(); c = Case("mixed"); p = c.pat
    sc = scen_mx(c.g); dr1 = c.staff()
    addr, pin = addr_en(c.loc, ch(["in", "none"]))
    adm = c.ref - timedelta(days=ri(2, 8))
    d.t(fac_en(c.loc, "big"), "\nDISCHARGE SUMMARY\n")
    F = [("IPD No", [E(ipd_no(), "MRN")]), ("Name", [hon_en(p), P(p)]), ("Age/Sex", [agesex_en(c)]), mx_guardian(c),
         ("Address", [E(addr, "ADDRESS")]), ("Mobile", [E(fmt_mobile(), "PHONE")])]
    if opt(0.4):
        F.append(("DOB", [E(fdate(c.dob), "DOB")]))
    if opt(0.4):
        F.append(("Aadhaar", [E(fmt_aadhaar(), "AADHAAR")]))
    if opt(0.3):
        F += abha_fields(c)
    if opt(0.25):
        F.append(idproof_field(p))
    F += [("DOA/DOD", [f"{fdate(adm)} / {fdate(c.ref)}"])]
    kv(d, F)
    rk = ch(["beta", "bhai", "damaad", "pati" if c.g == "F" else "biwi"])
    a = relative(p, "F" if rk == "biwi" else "M")
    d.t(f"\nDiagnosis: {sc['dx']}\n")
    d.t("History: ", P(p, "short"), f" ko {sc['cc']}. Unka {rk} ", P(a, ch(["short", "full"])), " emergency mein laaya. ")
    d.t(f"Investigations: {sc['lab']}. Course: IV fluids aur antibiotics diye, condition improve hui.\n")
    d.t("Discharge advice (ghar pe):\n")
    for r in sc["rx"]:
        d.t("- ", r, "\n")
    d.t(ch(["दवाई समय पर लें।", "Khana time pe khayein, tali hui cheezein band."]), f" Review on {fdate(c.ref + timedelta(days=7))}. {neg_number()}\n")
    d.t("Dr ", P(dr1), "\n")
    return "discharge_summary", d


def mx_tb():
    d = Doc(); c = Case("mixed"); p = c.pat
    sup = c.staff(pool=c.pool)
    addr, pin = addr_en(c.loc, ch(["in", "sep"]), hinglish=True)
    d.case = ch([None, "lower"])
    d.t("TB follow-up | Nikshay ID ", E(nikshay_id(), "MRN"), "\n")
    d.t(P(p), f" ({c.age}/{c.g}), ", E(addr, "ADDRESS"))
    if pin and not addr.endswith(pin):
        d.t(", pin ", E(pin, "PINCODE"))
    d.t(". Mob ", E(fmt_mobile(), "PHONE"), ".\n")
    w0 = ri(34, 50)
    d.t(f"Month {ri(2, 5)}: vajan {w0} se {w0 + ri(1, 4)} kg hua, khansi kam hai, sputum {ch(['negative', 'negative', 'pending'])}. ")
    d.t(f"Dawai regular le rahe hain ({ri(88, 100)}%). Nikshay Poshan ka paisa aa gaya. ")
    d.t("Treatment supporter ", P(sup, ch(["full", "short"])), ch([" (ASHA)", " (AWW)", ""]), ". ")
    fam = relative(p, ch("MF"))
    d.t("Ghar mein ", P(fam, "short"), f" ko bhi khansi hai - sputum sample liya ({neg_number()}).\n")
    return "tb_nikshay_followup", d


def mx_anc():
    d = Doc(); c = Case("mixed", g="F", age=(18, 35)); p = c.pat
    hus = relative(p, "M"); c._rel = ("husband", hus)
    anm = c.staff("F")
    addr, pin = addr_en(c.loc, ch(["in", "sep", "none"]), hinglish=True)
    lmp = c.ref - timedelta(days=ri(80, 240))
    d.t("ANC check-up - ", fac_en(c.loc, "small"), "\n")
    F = [("RCH ID", [E(rch_id(), "MRN")]), ("Naam", [P(p)]), ("Pati", [P(hus)]), ("Umar", [f"{c.age} saal"]),
         ("Pata", [E(addr, "ADDRESS")])]
    if pin and not addr.endswith(pin):
        F.append(("Pin", [E(pin, "PINCODE")]))
    F.append(("Mob", [E(fmt_mobile(), "PHONE")]))
    if opt(0.5):
        F.append(("Aadhar", [E(fmt_aadhaar(), "AADHAAR")]))
    if opt(0.3):
        F += abha_fields(c)
    if opt(0.25):
        F.append(("Voter ID", [E(epic_no(), "VOTER_ID")]))
    F += [("LMP", [fdate(lmp)]), ("EDD", [fdate(lmp + timedelta(days=280))])]
    kv(d, F)
    d.t(f"\nVajan {ri(42, 64)} kg, BP {ri(100, 150)}/{ri(64, 96)}, Hb {hb()}, FHR {ri(130, 152)}. ")
    d.t(ch(["पैरों में सूजन hai, urine albumin trace.", "Sab normal, IFA continue.", "Hb kam hai, iron sucrose ke liye refer."]), " ")
    d.t(f"Td-2 diya ({neg_number()}). ", P(p, "short"), " ko 108/102 number aur danger signs samjhaye. ANM ", P(anm, ch(["full", "short"])), "\n")
    return "anc_card", d


def mx_referral():
    d = Doc(); c = Case("mixed"); p = c.pat
    sc = scen_mx(c.g); mo = c.staff()
    addr, pin = addr_en(c.loc, ch(["in", "none"]), hinglish=True)
    d.t("Referral - ", fac_en(c.loc, "small"), " se ", fac_en(c.loc, "big"), f"\nDate {fdate(c.ref)}\n")
    d.t("Mareez ", P(p), f", {c.age} saal, ", mx_guardian(c)[0], " ", mx_guardian(c)[1], ", ", E(addr, "ADDRESS"), ". ")
    d.t(f"Problem: {mx_cc(sc)}. {sc['lab']}. Yahan {sc['rx'][0]} diya. Higher centre refer kiya ja raha hai.\n")
    rk = ch(["beta", "bhai", "pati" if c.g == "F" else "biwi"])
    a = relative(p, "F" if rk == "biwi" else "M")
    d.t(f"Saath mein {rk} ", P(a, ch(["short", "full"])), ", phone ", E(fmt_mobile(), "PHONE"), ". 108 se bheja.")
    if opt(0.4):
        d.t(" Patient ka ", ch(["voter card ", "Voter ID "]), E(epic_no(), "VOTER_ID"), " copy attached.")
    d.t("\n")
    d.t(dr_en(), P(mo), ", MO\n")
    return "referral_letter", d


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
EN_T = [en_discharge, en_opd, en_referral, en_asha, en_lab, en_anc, en_tb, en_imm, en_death, en_birth, en_tele, en_pmjay]
HI_T = [hi_opd, hi_discharge, hi_referral, hi_asha, hi_anc, hi_tb, hi_imm, hi_birth, hi_death, hi_pmjay]
MX_T = [mx_tele, mx_asha, mx_opd, mx_discharge, mx_tb, mx_anc, mx_referral]

PAT = {
    "AADHAAR": None, "PAN": re.compile(r"^[A-Z]{3}P[A-Z]\d{4}[A-Z]$"), "VOTER_ID": re.compile(r"^[A-Z]{3}\d{7}$"),
    "ABHA_ADDRESS": re.compile(r"^[a-z0-9._]+@(abdm|sbx)$"), "EMAIL": re.compile(r"^[a-z0-9._]+@[a-z.]+$"),
    "PINCODE": re.compile(r"^\d{6}$"),
}


import unicodedata


def _isword(chr_):
    return bool(chr_) and unicodedata.category(chr_)[0] in "LMN"


def unlabelled_mentions(rec):
    """PERSON strings that also occur elsewhere in the note outside any entity span."""
    text, ents = rec["text"], rec["entities"]
    cov = [False] * len(text)
    for e in ents:
        for i in range(e["start"], e["end"]):
            cov[i] = True
    out = []
    for name in sorted({e["text"] for e in ents if e["type"] == "PERSON"}):
        for m in re.finditer(re.escape(name), text):
            a, b = m.start(), m.end()
            if any(cov[a:b]):
                continue
            if _isword(text[a - 1] if a else "") or _isword(text[b] if b < len(text) else ""):
                continue
            out.append((name, a))
    return out


def validate(rec):
    text = rec["text"]
    last_end = -1
    for e in sorted(rec["entities"], key=lambda x: x["start"]):
        assert text[e["start"]:e["end"]] == e["text"], (rec["id"], e)
        assert e["text"] and e["text"] == e["text"].strip(), (rec["id"], e)
        assert e["start"] >= last_end, ("overlap", rec["id"], e)
        last_end = e["end"]
        t = e["type"]
        v = e["text"].translate(LAT_TR)
        if t == "AADHAAR":
            dg = re.sub(r"[ -]", "", v)
            if re.fullmatch(r"\d{12}", dg):
                assert dg[0] in "23456789" and verhoeff_ok(dg), (rec["id"], e)
            else:
                assert re.fullmatch(r"[Xx*]{8}\d{4}", dg), (rec["id"], e)
        elif t == "ABHA_NUMBER":
            assert re.fullmatch(r"\d{14}", re.sub(r"[ -]", "", v)), (rec["id"], e)
        elif t == "PHONE":
            dg = re.sub(r"\D", "", v)
            assert 10 <= len(dg) <= 12, (rec["id"], e)
        elif t in PAT and PAT[t] is not None:
            assert PAT[t].match(v), (rec["id"], e)
        elif t == "MRN":
            dg = re.sub(r"\D", "", v)
            if len(dg) == 12 and dg == re.sub(r"[ -]", "", v):
                assert not verhoeff_ok(dg) or dg[0] in "01", ("MRN looks like Aadhaar", rec["id"], e)


def main():
    plan = []
    for i in range(N_EN):
        plan.append(("en", EN_T[i % len(EN_T)]))
    for i in range(N_HI):
        plan.append(("hi", HI_T[i % len(HI_T)]))
    for i in range(N_MX):
        plan.append(("mixed", MX_T[i % len(MX_T)]))
    R.shuffle(plan)
    recs = []
    for k, (lang, fn) in enumerate(plan, 1):
        dtype, d = fn()
        rec = {"id": f"ho-{k:04d}", "lang": lang, "doc_type": dtype, "text": d.text(), "entities": d.ents}
        validate(rec)
        um = unlabelled_mentions(rec)
        assert not um, (rec["id"], um)
        recs.append(rec)
    with open(OUT, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_lang = Counter(r["lang"] for r in recs)
    by_type = Counter(e["type"] for r in recs for e in r["entities"])
    by_lang_type = Counter((r["lang"], e["type"]) for r in recs for e in r["entities"])
    by_doc = Counter(r["doc_type"] for r in recs)
    print(f"wrote {len(recs)} notes -> {OUT}")
    print("per lang:", dict(sorted(by_lang.items())))
    print("per doc_type:", dict(sorted(by_doc.items())))
    print("per entity type (total / en / hi / mixed):")
    for t in ["PERSON", "ADDRESS", "AADHAAR", "ABHA_NUMBER", "ABHA_ADDRESS", "PAN", "VOTER_ID", "PHONE", "EMAIL",
              "MRN", "DOB", "PINCODE"]:
        print(f"  {t:<13} {by_type[t]:>4}  /  {by_lang_type[('en', t)]:>4} {by_lang_type[('hi', t)]:>4} {by_lang_type[('mixed', t)]:>4}")
    print("total entities:", sum(by_type.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
