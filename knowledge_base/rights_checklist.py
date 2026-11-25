"""Women at Workplace rights checklist.

Each rule is a short, plain-language summary of a legal right, with the Act and
section it comes from. The Women at Workplace agent compares an uploaded document
against these rules. The full Act text (from India Code PDFs) is searched
separately so answers can quote the actual law.

Status of law: the four Labour Codes (incl. Code on Social Security, 2020 and
Code on Wages, 2019) came into force on 21 November 2025. They replaced the
Maternity Benefit Act, 1961 and the Equal Remuneration Act, 1976.
"""

SS = "Code on Social Security, 2020"
WAGES = "Code on Wages, 2019"
POSH = "Sexual Harassment of Women at Workplace (Prevention, Prohibition and Redressal) Act, 2013"
CONTRACT = "Indian Contract Act, 1872"
DPDP = "Digital Personal Data Protection Act, 2023"

RULES = [
    # ---------------- Maternity ----------------
    dict(id="MAT-1", topic="maternity", act=SS, section="60",
         title="Maternity leave of 26 weeks",
         rule="A woman employee is entitled to paid maternity benefit for up to 26 weeks (not more than 8 weeks before the expected delivery). "
              "For a woman with two or more surviving children it is 12 weeks. She must have worked at least 80 days in the 12 months before the expected delivery.",
         look_for="maternity leave duration; any period less than 26 weeks for first two children is a problem",
         keywords=["maternity", "pregnan", "delivery", "childbirth", "maternity leave"]),
    dict(id="MAT-2", topic="maternity", act=SS, section="60",
         title="Adoption and commissioning mothers",
         rule="A woman who legally adopts a child, or a commissioning mother (surrogacy), is entitled to 12 weeks of maternity benefit. "
              "(The Supreme Court has struck down the old condition that the adopted child must be below 3 months.)",
         look_for="adoption / surrogacy leave missing or less than 12 weeks",
         keywords=["adopt", "surrogacy", "commissioning mother"]),
    dict(id="MAT-3", topic="maternity", act=SS, section="65",
         title="Leave for miscarriage",
         rule="In case of miscarriage or medical termination of pregnancy, a woman is entitled to 6 weeks of paid leave after the event.",
         look_for="no miscarriage leave, or less than 6 weeks",
         keywords=["miscarriage", "termination of pregnancy"]),
    dict(id="MAT-4", topic="maternity", act=SS, section="66",
         title="Nursing breaks",
         rule="After returning to work, a mother is entitled to two nursing breaks a day (in addition to the normal rest interval) until the child is 15 months old.",
         look_for="nursing breaks denied or not mentioned",
         keywords=["nursing", "breastfeed", "feeding"]),
    dict(id="MAT-5", topic="maternity", act=SS, section="67",
         title="Crèche facility (50+ employees)",
         rule="Every establishment with 50 or more employees must provide a crèche facility (alone or shared), and the mother may visit it four times a day. "
              "Where the nature of work allows, the employer may let her work from home after maternity leave on mutually agreed terms.",
         look_for="no crèche in a company with 50+ employees; work-from-home after maternity refused outright",
         keywords=["creche", "crèche", "childcare", "day care", "work from home"]),
    dict(id="MAT-6", topic="maternity", act=SS, section="68",
         title="No dismissal during maternity absence",
         rule="An employer cannot dismiss or discharge a woman during, or on account of, her absence on maternity leave, or change her conditions of service to her disadvantage during that time.",
         look_for="termination for long absence that would include maternity leave; probation/pregnancy-based termination",
         keywords=["terminat", "dismiss", "absence", "pregnan"]),
    dict(id="MAT-7", topic="maternity", act=SS, section="69",
         title="No wage cut because of maternity",
         rule="No deduction can be made from a woman's normal wages because of the nature of work assigned during pregnancy or because of nursing breaks.",
         look_for="pay cut or reduced bonus/increment due to maternity",
         keywords=["deduct", "maternity", "increment", "bonus"]),
    # ---------------- POSH ----------------
    dict(id="POSH-1", topic="posh", act=POSH, section="4",
         title="Internal Committee is mandatory (10+ employees)",
         rule="Every employer with 10 or more employees must form an Internal Committee (IC) for sexual harassment complaints. "
              "It is headed by a senior woman employee, at least half its members must be women, and one member must be from an outside NGO or a person familiar with women's issues.",
         look_for="no Internal Committee; complaints only to manager/HR head; committee without external member",
         keywords=["harass", "internal committee", "complaint", "posh", "icc"]),
    dict(id="POSH-2", topic="posh", act=POSH, section="6",
         title="Local Committee for small workplaces",
         rule="Where an establishment has fewer than 10 employees (or the complaint is against the employer), the woman can complain to the Local Committee set up by the District Officer.",
         look_for="small office with no complaint route",
         keywords=["local committee", "district"]),
    dict(id="POSH-3", topic="posh", act=POSH, section="9",
         title="Time to file a complaint",
         rule="A complaint of sexual harassment can be made in writing within 3 months of the incident (or the last incident). The committee can extend this by another 3 months for valid reasons.",
         look_for="shorter deadline than 3 months, or rules discouraging complaints",
         keywords=["complaint", "within", "days", "harass"]),
    dict(id="POSH-4", topic="posh", act=POSH, section="11",
         title="Inquiry within 90 days",
         rule="The Internal Committee must complete its inquiry within 90 days.",
         look_for="no inquiry timeline",
         keywords=["inquiry", "investigation"]),
    dict(id="POSH-5", topic="posh", act=POSH, section="12",
         title="Relief during the inquiry",
         rule="During the inquiry, on the woman's written request, the committee can recommend her transfer or the respondent's transfer, or leave of up to 3 months for her (in addition to her normal leave).",
         look_for="no interim relief",
         keywords=["transfer", "interim", "leave"]),
    dict(id="POSH-6", topic="posh", act=POSH, section="16",
         title="Confidentiality",
         rule="The identity of the complainant, respondent and witnesses, and the inquiry details, must be kept confidential.",
         look_for="policy allowing disclosure of complainant identity",
         keywords=["confidential", "identity", "disclos"]),
    dict(id="POSH-7", topic="posh", act=POSH, section="19",
         title="Employer's duties",
         rule="The employer must provide a safe workplace, display the consequences of sexual harassment and the IC details at the workplace, hold awareness workshops, and assist the woman if she wants to file a police complaint.",
         look_for="no POSH policy, no display of IC details, no awareness programme",
         keywords=["policy", "display", "awareness", "safe"]),
    dict(id="POSH-8", topic="posh", act=POSH, section="10",
         title="Conciliation cannot be a money settlement",
         rule="Conciliation is only at the woman's request, and no monetary settlement can be the basis of conciliation.",
         look_for="forced mediation or settlement by payment",
         keywords=["settle", "mediat", "conciliat"]),
    dict(id="POSH-9", topic="posh", act=POSH, section="26",
         title="Penalty on employer",
         rule="An employer who fails to form the Internal Committee or follow the Act can be fined up to ₹50,000; repeat violations can lead to double punishment and cancellation of licence/registration.",
         look_for="(information)",
         keywords=["penalty", "fine"]),
    # ---------------- Equal pay & wages ----------------
    dict(id="PAY-1", topic="equal_pay", act=WAGES, section="3",
         title="Equal pay, no gender discrimination",
         rule="An employer cannot discriminate on the ground of gender (including transgender identity) in wages for the same or similar work, "
              "or in recruitment and conditions of employment for such work.",
         look_for="different pay, allowance or bonus for women for the same role",
         keywords=["female", "women", "gender", "equal", "allowance", "salary", "pay"]),
    dict(id="PAY-2", topic="wages", act=WAGES, section="5",
         title="Minimum wages",
         rule="No employer can pay an employee less than the minimum rate of wages notified by the government.",
         look_for="pay below minimum wage",
         keywords=["minimum wage", "salary", "stipend"]),
    dict(id="PAY-3", topic="wages", act=WAGES, section="17",
         title="Timely payment of wages",
         rule="Monthly wages must be paid before the 7th day of the next month. If an employee resigns or is removed, final dues must be paid within 2 working days.",
         look_for="salary paid after the 7th of the next month; full and final settlement delayed for 45/60/90 days",
         keywords=["salary", "paid", "payment", "full and final", "settlement", "dues"]),
    dict(id="PAY-4", topic="wages", act=WAGES, section="14",
         title="Overtime at double rate",
         rule="Work beyond normal working hours must be paid as overtime at not less than twice the normal rate of wages.",
         look_for="no overtime pay, or overtime at normal rate",
         keywords=["overtime", "extra hours", "working hours"]),
    dict(id="PAY-5", topic="wages", act=WAGES, section="18",
         title="Limits on deductions",
         rule="Only deductions allowed by law can be made from wages, and total deductions cannot exceed 50% of the wages.",
         look_for="penalties/deductions that are unlimited or more than half the salary",
         keywords=["deduct", "fine", "penalty", "recover"]),
    # ---------------- Contract terms ----------------
    dict(id="CON-1", topic="contract", act=CONTRACT, section="27",
         title="Non-compete after leaving is generally void",
         rule="An agreement that stops a person from carrying on a lawful profession, trade or business is void. Indian courts usually do not enforce non-compete clauses that apply after the job ends.",
         look_for="non-compete restriction after resignation/termination",
         keywords=["non-compete", "compete", "restrain", "competitor", "after leaving", "post-employment"]),
    dict(id="CON-2", topic="contract", act=CONTRACT, section="74",
         title="Bonds and penalties must be reasonable",
         rule="If a contract names a penalty or fixed amount for breach (for example a training bond), the employer can recover only reasonable compensation for actual loss, not more than the amount named.",
         look_for="large training bond / notice-period penalty unconnected to real cost",
         keywords=["bond", "penalty", "liquidated", "damages", "training cost", "compensation"]),
    dict(id="CON-3", topic="contract", act=CONTRACT, section="14",
         title="Free consent",
         rule="Consent obtained by coercion, undue influence, fraud or misrepresentation is not free consent, and such a contract can be cancelled by the affected person.",
         look_for="forced signing, documents withheld, coercive terms",
         keywords=["consent", "coerc", "pressure", "force"]),
    dict(id="CON-4", topic="contract", act=CONTRACT, section="23",
         title="Unlawful terms are void",
         rule="A term whose object is unlawful or against public policy (for example, waiving legal rights such as maternity benefit) is void.",
         look_for="employee waives statutory rights",
         keywords=["waive", "forgo", "unlawful", "public policy"]),
    # ---------------- Data privacy ----------------
    dict(id="DATA-1", topic="privacy", act=DPDP, section="6",
         title="Consent for personal data",
         rule="Personal data should be processed for a specified, lawful purpose. Consent must be free, specific, informed and limited to the data needed; it can be withdrawn. "
              "Employers may process data needed for employment purposes without fresh consent (section 7), but blanket sharing with any third party is not a 'specified purpose'.",
         look_for="consent to share personal data with any third party for any purpose; no way to withdraw",
         keywords=["personal data", "privacy", "third part", "share", "data", "consent"]),
    dict(id="DATA-2", topic="privacy", act=DPDP, section="12",
         title="Right to correction and erasure",
         rule="A person can ask for their personal data to be corrected, completed, updated or erased when it is no longer needed for the purpose.",
         look_for="company keeps data forever / refuses deletion",
         keywords=["retain", "erase", "delete", "forever"]),
    # ---------------- Rent (Maharashtra focus, as the team is in Pune) ----------------
    dict(id="RENT-1", topic="rent", act="Maharashtra Rent Control Act, 1999", section="55",
         title="Leave and licence agreements must be registered (Maharashtra)",
         rule="In Maharashtra, a leave and licence or rent agreement must be in writing and registered, and the landlord is responsible for getting it registered. "
              "If it is not registered, the tenant's version of the terms is accepted in a dispute.",
         look_for="agreement says it will not be registered",
         keywords=["register", "stamp", "unregistered"]),
    dict(id="RENT-2", topic="rent", act="Model Tenancy Act, 2021", section="11",
         title="Security deposit limit (model law)",
         rule="The central Model Tenancy Act, 2021 suggests a security deposit of at most 2 months' rent for residential premises, refunded when the tenant leaves after lawful deductions. "
              "It applies only in states that adopt it, but it is a useful fairness benchmark.",
         look_for="deposit much higher than 2 months; refund delayed or at landlord's discretion",
         keywords=["deposit", "refund", "forfeit"]),
]

RULES_BY_ID = {r["id"]: r for r in RULES}

TOPIC_LABELS = {
    "maternity": "Maternity benefits",
    "posh": "Sexual harassment (POSH)",
    "equal_pay": "Equal pay",
    "wages": "Wages",
    "contract": "Contract terms",
    "privacy": "Data privacy",
    "rent": "Rent agreement",
}

WOMEN_TOPICS = {"maternity", "posh", "equal_pay"}


def rules_for_topics(topics) -> list[dict]:
    topics = set(topics or [])
    if not topics:
        return RULES
    return [r for r in RULES if r["topic"] in topics]


def rule_as_text(r: dict) -> str:
    return f"[{r['id']}] {r['title']} — {r['act']}, Section {r['section']}: {r['rule']}"
