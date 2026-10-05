"""The bundled cases: real, openly licensed laparoscopic and open-surgery clips.

`procedure` is what the clip's own source article says is happening. It is the one
piece of context the operator supplies; Scopewatch cannot recognise anatomy, so it
is passed to the reasoning layer as a stated fact, never as something the pixels showed.
"""
from __future__ import annotations

CASES = [
    {
        "slug": "boer-cholecystectomy",
        "title": "Cholecystectomy, gallbladder torsion",
        "blurb": "Dissection of the cystic duct and artery, bleeding at the gallbladder bed.",
        "procedure": ("Laparoscopic cholecystectomy for a torted, gangrenous gallbladder: "
                      "derotation, then dissection of the cystic duct and artery, with "
                      "bleeding at the gallbladder bed (source article's description)."),
        "credit": "Boer J, Boerma D, de Vries Reilingh T. J Med Case Rep 2011, doi:10.1186/1752-1947-5-588",
        "licence": "CC BY 2.0",
        "kind": "laparoscopic",
    },
    {
        "slug": "wses-bleeding-ulcer",
        "title": "Bleeding peptic ulcer repair",
        "blurb": "Suture repair with blood pooling in the gutter throughout.",
        "procedure": ("Laparoscopic suture repair of a perforated, bleeding peptic ulcer "
                      "(source article's description). Dark blood is already pooled in the "
                      "gutter from the first surgical frame."),
        "credit": "Di Saverio S et al. World J Emerg Surg 2014, doi:10.1186/1749-7922-9-45",
        "licence": "CC BY 4.0",
        "kind": "laparoscopic",
    },
    {
        "slug": "kaplan-pancreatic-dissection",
        "title": "Pancreatic and bile duct dissection",
        "blurb": "A long dissection with fresh bleeding near the end.",
        "procedure": ("Total laparoscopic pancreaticoduodenectomy, the technique video of "
                      "a case report: adhesiolysis, then dissection of the pancreas and the "
                      "common bile duct (source article's description)."),
        "credit": "Kaplan M. World J Surg Oncol 2012, doi:10.1186/1477-7819-10-142",
        "licence": "CC BY 2.0",
        "kind": "laparoscopic",
    },
    {
        "slug": "barroso-inguinal-suture",
        "title": "Inguinal ring suture, paediatric",
        "blurb": "A clean preperitoneal view; the quiet case.",
        "procedure": ("Laparoscopic closure of the internal inguinal ring in a child "
                      "(source article's description). A clean field with a few "
                      "needle-puncture spots."),
        "credit": "Barroso C et al. Front Pediatr 2017, doi:10.3389/fped.2017.00207",
        "licence": "CC BY 4.0",
        "kind": "laparoscopic",
    },
    {
        "slug": "gupta-open-surgery",
        "title": "Open surgery (a refusal)",
        "blurb": "Not a laparoscopic view. Scopewatch is built to say so.",
        "procedure": ("OPEN cholecystectomy for gallbladder torsion (source article's "
                      "description): gloved hands in an open abdomen, not a laparoscope."),
        "credit": "Gupta V et al. Cases J 2009, doi:10.1186/1757-1626-2-193",
        "licence": "CC BY 2.0",
        "kind": "open",
    },
]
BY_SLUG = {c["slug"]: c for c in CASES}
