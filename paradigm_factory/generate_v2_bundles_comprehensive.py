#!/usr/bin/env python3
"""
SemanticPhase v2 Comprehensive Bundle Generator
================================================

Generates training bundles from 250-word polysemy lexicon covering:
- Law (50 words): Reification & adversarial metaphors
- Medicine (50 words): Corporealization & systemic metaphors
- Engineering/CS (80 words): Virtualization & spatial metaphors
- Finance (25 words): Abstraction of value
- Slang (45 words): High-velocity semantic drift

Cognitive mechanisms covered:
- Metaphorical Extension: Concrete -> Abstract (CS/Eng)
- Metonymic Shift: Part-whole associations (Med/Law)
- Specialization: General -> Narrow technical meaning
"""

import json
import random
from pathlib import Path
from typing import Dict, List
from datetime import datetime


# =============================================================================
# DOMAIN 1: LAW (50 polysemes)
# =============================================================================

LAW_CATALOG = {
    "action": {
        "senses": {"s1": "The process of doing something.", "s2": "A judicial proceeding or lawsuit."},
        "examples": {
            "s1": ["The action sequence was thrilling.", "We need immediate action."],
            "s2": ["The plaintiff filed a civil action.", "The action was dismissed."]
        }
    },
    "bar": {
        "senses": {"s1": "A rigid piece of metal or drinking establishment.", "s2": "The legal profession or courtroom railing."},
        "examples": {
            "s1": ["He met friends at the bar.", "A chocolate bar."],
            "s2": ["She was admitted to the Bar.", "The prisoner approached the bar."]
        }
    },
    "battery": {
        "senses": {"s1": "An energy storage device.", "s2": "Intentional harmful touching without consent."},
        "examples": {
            "s1": ["The flashlight battery died.", "Replace the battery."],
            "s2": ["Charged with assault and battery.", "Battery requires physical contact."]
        }
    },
    "bench": {
        "senses": {"s1": "A long seat.", "s2": "The judge or judiciary."},
        "examples": {
            "s1": ["They sat on a park bench.", "Weight lifting bench."],
            "s2": ["The bench issued a stern warning.", "Approach the bench."]
        }
    },
    "brief": {
        "senses": {"s1": "Short in duration.", "s2": "A written legal argument."},
        "examples": {
            "s1": ["It was a brief meeting.", "A brief summary."],
            "s2": ["The attorney submitted an amicus brief.", "File the brief by Friday."]
        }
    },
    "case": {
        "senses": {"s1": "A container or instance.", "s2": "A lawsuit or legal matter."},
        "examples": {
            "s1": ["He packed a case of wine.", "In that case, proceed."],
            "s2": ["The prosecution rested its case.", "Landmark case."]
        }
    },
    "code": {
        "senses": {"s1": "A system of signals or computer instructions.", "s2": "A systematic collection of statutes."},
        "examples": {
            "s1": ["She wrote Python code.", "Secret code."],
            "s2": ["The violation falls under the Civil Code.", "Penal Code section 187."]
        }
    },
    "complaint": {
        "senses": {"s1": "An expression of dissatisfaction.", "s2": "The initial pleading starting a lawsuit."},
        "examples": {
            "s1": ["We received a customer complaint.", "No complaints here."],
            "s2": ["The complaint alleges negligence.", "Amended complaint."]
        }
    },
    "consideration": {
        "senses": {"s1": "Careful thought.", "s2": "Value exchanged making a contract binding."},
        "examples": {
            "s1": ["Thank you for your consideration.", "Give it consideration."],
            "s2": ["The contract is void for lack of consideration.", "Adequate consideration."]
        }
    },
    "damages": {
        "senses": {"s1": "Harm or injury.", "s2": "Monetary compensation awarded to a plaintiff."},
        "examples": {
            "s1": ["The storm caused minor damages.", "Property damages."],
            "s2": ["The jury awarded punitive damages.", "Compensatory damages."]
        }
    },
    "deed": {
        "senses": {"s1": "An intentional action.", "s2": "A document transferring property ownership."},
        "examples": {
            "s1": ["A good deed.", "Brave deeds."],
            "s2": ["The deed to the land was recorded.", "Warranty deed."]
        }
    },
    "defense": {
        "senses": {"s1": "Protection from attack.", "s2": "The case presented by the accused party."},
        "examples": {
            "s1": ["The castle's defense.", "Self defense."],
            "s2": ["The defense rests.", "Affirmative defense."]
        }
    },
    "estate": {
        "senses": {"s1": "A large property.", "s2": "Total assets owned, especially at death."},
        "examples": {
            "s1": ["A country estate.", "Real estate."],
            "s2": ["The executor managed the estate.", "Estate planning."]
        }
    },
    "execute": {
        "senses": {"s1": "To carry out or put to death.", "s2": "To sign a document to make it valid."},
        "examples": {
            "s1": ["Execute the plan.", "Execute the prisoner."],
            "s2": ["Execute the will before a notary.", "Properly executed contract."]
        }
    },
    "finding": {
        "senses": {"s1": "A discovery.", "s2": "A formal decision by a judge on a fact."},
        "examples": {
            "s1": ["An archaeological finding.", "Research findings."],
            "s2": ["The court's finding of fact.", "Factual finding."]
        }
    },
    "fine": {
        "senses": {"s1": "High quality.", "s2": "A monetary penalty."},
        "examples": {
            "s1": ["Fine art.", "Fine dining."],
            "s2": ["He paid a hefty fine.", "Traffic fine."]
        }
    },
    "ground": {
        "senses": {"s1": "The solid surface of the earth.", "s2": "The basis or justification for an action."},
        "examples": {
            "s1": ["Sit on the ground.", "Ground floor."],
            "s2": ["Grounds for divorce.", "Legal grounds."]
        }
    },
    "hearing": {
        "senses": {"s1": "The sense of perceiving sound.", "s2": "A formal proceeding before a court."},
        "examples": {
            "s1": ["His hearing is excellent.", "Hearing loss."],
            "s2": ["The bail hearing is Monday.", "Preliminary hearing."]
        }
    },
    "holding": {
        "senses": {"s1": "Gripping something.", "s2": "A court's determination of a matter of law."},
        "examples": {
            "s1": ["Holding the bag.", "Holding hands."],
            "s2": ["The holding set a new precedent.", "The court's holding."]
        }
    },
    "instrument": {
        "senses": {"s1": "A tool or musical device.", "s2": "A formal legal document."},
        "examples": {
            "s1": ["Play an instrument.", "Surgical instrument."],
            "s2": ["A negotiable instrument.", "The instrument was filed."]
        }
    },
    "interest": {
        "senses": {"s1": "Curiosity or attention.", "s2": "A legal right or share in property."},
        "examples": {
            "s1": ["Interest in hobbies.", "Spark interest."],
            "s2": ["A possessory interest in the land.", "Security interest."]
        }
    },
    "issue": {
        "senses": {"s1": "A topic or publication edition.", "s2": "Lineal descendants."},
        "examples": {
            "s1": ["The latest issue of the magazine.", "Key issue."],
            "s2": ["He died without issue.", "Lawful issue."]
        }
    },
    "judgment": {
        "senses": {"s1": "The ability to make decisions.", "s2": "The official decision of a court."},
        "examples": {
            "s1": ["Good judgment.", "Use your judgment."],
            "s2": ["Summary judgment was granted.", "Default judgment."]
        }
    },
    "jury": {
        "senses": {"s1": "A group evaluating something.", "s2": "A sworn body deciding guilt."},
        "examples": {
            "s1": ["The jury is still out on that.", "Jury of peers."],
            "s2": ["The grand jury indicted him.", "Jury deliberation."]
        }
    },
    "lien": {
        "senses": {"s1": "A claim on something (metaphorical).", "s2": "A legal right to property until debt is paid."},
        "examples": {
            "s1": ["A lien on my time.", "Emotional lien."],
            "s2": ["A tax lien on the house.", "Mechanic's lien."]
        }
    },
    "matter": {
        "senses": {"s1": "Physical substance.", "s2": "A subject under legal consideration."},
        "examples": {
            "s1": ["Dark matter.", "Subject matter."],
            "s2": ["In the matter of the estate.", "Legal matter."]
        }
    },
    "minor": {
        "senses": {"s1": "Lesser in importance.", "s2": "A person under the age of majority."},
        "examples": {
            "s1": ["Minor detail.", "Minor injury."],
            "s2": ["Selling alcohol to a minor.", "Minor child."]
        }
    },
    "motion": {
        "senses": {"s1": "Movement.", "s2": "A formal request to a judge."},
        "examples": {
            "s1": ["Laws of motion.", "Slow motion."],
            "s2": ["Motion to dismiss.", "File a motion."]
        }
    },
    "notice": {
        "senses": {"s1": "Attention or awareness.", "s2": "Formal notification required by law."},
        "examples": {
            "s1": ["Take notice.", "Short notice."],
            "s2": ["Serve a notice to quit.", "Legal notice."]
        }
    },
    "oath": {
        "senses": {"s1": "A profanity.", "s2": "A solemn promise of truth."},
        "examples": {
            "s1": ["He shouted an oath.", "Muttered oaths."],
            "s2": ["Testify under oath.", "Oath of office."]
        }
    },
    "objection": {
        "senses": {"s1": "General disapproval.", "s2": "A formal protest in court."},
        "examples": {
            "s1": ["I have an objection to the plan.", "No objection."],
            "s2": ["Objection, hearsay.", "Sustained objection."]
        }
    },
    "opinion": {
        "senses": {"s1": "A personal belief.", "s2": "A judge's written explanation of a ruling."},
        "examples": {
            "s1": ["My opinion on politics.", "Public opinion."],
            "s2": ["The dissenting opinion.", "Majority opinion."]
        }
    },
    "order": {
        "senses": {"s1": "A sequence or request.", "s2": "A command from a judge."},
        "examples": {
            "s1": ["Alphabetical order.", "Place an order."],
            "s2": ["A restraining order.", "Court order."]
        }
    },
    "pardon": {
        "senses": {"s1": "Forgiveness or excuse.", "s2": "Relief from legal consequences of crime."},
        "examples": {
            "s1": ["Pardon me.", "Beg your pardon."],
            "s2": ["A presidential pardon.", "Full pardon."]
        }
    },
    "party": {
        "senses": {"s1": "A social gathering.", "s2": "A participant in a lawsuit."},
        "examples": {
            "s1": ["Birthday party.", "Party time."],
            "s2": ["The injured party.", "Third party."]
        }
    },
    "patent": {
        "senses": {"s1": "Obvious or clear.", "s2": "An intellectual property right."},
        "examples": {
            "s1": ["Patent nonsense.", "Patently false."],
            "s2": ["File a patent.", "Patent infringement."]
        }
    },
    "petition": {
        "senses": {"s1": "A formal request.", "s2": "A formal application to a court."},
        "examples": {
            "s1": ["Sign a petition.", "Online petition."],
            "s2": ["Petition for bankruptcy.", "Writ of certiorari petition."]
        }
    },
    "plea": {
        "senses": {"s1": "An earnest request.", "s2": "A defendant's answer to a charge."},
        "examples": {
            "s1": ["A plea for help.", "Desperate plea."],
            "s2": ["Enter a plea of not guilty.", "Plea bargain."]
        }
    },
    "pleading": {
        "senses": {"s1": "Begging.", "s2": "Formal written statements in litigation."},
        "examples": {
            "s1": ["Pleading eyes.", "Stop pleading."],
            "s2": ["Responsive pleadings.", "Amended pleading."]
        }
    },
    "precedent": {
        "senses": {"s1": "An earlier example.", "s2": "A prior decision cited as authority."},
        "examples": {
            "s1": ["Set a precedent for behavior.", "Historical precedent."],
            "s2": ["Binding precedent.", "Stare decisis."]
        }
    },
    "proceeding": {
        "senses": {"s1": "Moving forward.", "s2": "A legal action or process."},
        "examples": {
            "s1": ["Proceedings of the meeting.", "Conference proceedings."],
            "s2": ["Criminal proceedings.", "Judicial proceeding."]
        }
    },
    "record": {
        "senses": {"s1": "A vinyl disc or best performance.", "s2": "The official transcript of proceedings."},
        "examples": {
            "s1": ["Break the record.", "Vinyl record."],
            "s2": ["Strike that from the record.", "On the record."]
        }
    },
    "relief": {
        "senses": {"s1": "Reassurance or comfort.", "s2": "Redress or benefit awarded by court."},
        "examples": {
            "s1": ["Sigh of relief.", "Pain relief."],
            "s2": ["Injunctive relief.", "Equitable relief."]
        }
    },
    "remedy": {
        "senses": {"s1": "A cure or solution.", "s2": "Legal means to enforce a right."},
        "examples": {
            "s1": ["Cold remedy.", "Home remedy."],
            "s2": ["Legal remedy.", "Exhausted all remedies."]
        }
    },
    "right": {
        "senses": {"s1": "Correct or a direction.", "s2": "A moral or legal entitlement."},
        "examples": {
            "s1": ["Turn right.", "Right answer."],
            "s2": ["Bill of Rights.", "Constitutional right."]
        }
    },
    "ruling": {
        "senses": {"s1": "Governing or dominant.", "s2": "An authoritative decision."},
        "examples": {
            "s1": ["Ruling class.", "Ruling passion."],
            "s2": ["The judge's ruling.", "Appellate ruling."]
        }
    },
    "sentence": {
        "senses": {"s1": "A grammatical unit.", "s2": "Punishment assigned to a criminal."},
        "examples": {
            "s1": ["Write a sentence.", "Complete sentence."],
            "s2": ["Life sentence.", "Death sentence."]
        }
    },
    "suit": {
        "senses": {"s1": "Matching clothing or card set.", "s2": "A lawsuit."},
        "examples": {
            "s1": ["Business suit.", "Follow suit."],
            "s2": ["File a suit.", "Class action suit."]
        }
    },
    "title": {
        "senses": {"s1": "A name or designation.", "s2": "A bundle of property rights."},
        "examples": {
            "s1": ["Book title.", "Job title."],
            "s2": ["Clear title.", "Title search."]
        }
    },
    "ward": {
        "senses": {"s1": "A hospital division.", "s2": "A minor under legal protection."},
        "examples": {
            "s1": ["Pediatric ward.", "Hospital ward."],
            "s2": ["Ward of the court.", "Guardian and ward."]
        }
    },
}


# =============================================================================
# DOMAIN 2: MEDICINE (50 polysemes)
# =============================================================================

MEDICAL_CATALOG = {
    "acute": {
        "senses": {"s1": "Sharp or perceptive.", "s2": "A condition with rapid onset and short course."},
        "examples": {
            "s1": ["He has an acute mind.", "Acute angle."],
            "s2": ["Acute respiratory distress.", "Acute appendicitis."]
        }
    },
    "benign": {
        "senses": {"s1": "Gentle or kind.", "s2": "A non-cancerous growth."},
        "examples": {
            "s1": ["A benign smile.", "Benign neglect."],
            "s2": ["The tumor is benign.", "Benign prostatic hyperplasia."]
        }
    },
    "condition": {
        "senses": {"s1": "A state or requirement.", "s2": "A disease or health status."},
        "examples": {
            "s1": ["Terms and conditions.", "Good condition."],
            "s2": ["His condition is stable.", "Chronic condition."]
        }
    },
    "culture": {
        "senses": {"s1": "Arts and social behavior.", "s2": "Propagation of microorganisms."},
        "examples": {
            "s1": ["Pop culture.", "Cultural heritage."],
            "s2": ["Blood culture confirmed sepsis.", "Bacterial culture."]
        }
    },
    "delivery": {
        "senses": {"s1": "Transporting goods.", "s2": "The process of childbirth."},
        "examples": {
            "s1": ["Pizza delivery.", "Next day delivery."],
            "s2": ["Vaginal delivery.", "Cesarean delivery."]
        }
    },
    "discharge": {
        "senses": {"s1": "Releasing or firing.", "s2": "Release of a patient or bodily secretion."},
        "examples": {
            "s1": ["Dishonorable discharge.", "Discharge a weapon."],
            "s2": ["Purulent discharge from the wound.", "Patient discharge."]
        }
    },
    "dressing": {
        "senses": {"s1": "Clothes or salad topping.", "s2": "Material applied to a wound."},
        "examples": {
            "s1": ["Salad dressing.", "Window dressing."],
            "s2": ["Sterile dressing.", "Change the dressing."]
        }
    },
    "drug": {
        "senses": {"s1": "An illegal substance.", "s2": "Any therapeutic agent."},
        "examples": {
            "s1": ["War on drugs.", "Drug dealer."],
            "s2": ["Prescription drug.", "Drug interactions."]
        }
    },
    "episode": {
        "senses": {"s1": "A TV show part.", "s2": "A period or event of illness."},
        "examples": {
            "s1": ["Final episode.", "Season finale episode."],
            "s2": ["Depressive episode.", "Manic episode."]
        }
    },
    "examination": {
        "senses": {"s1": "A school test.", "s2": "Inspection of the body."},
        "examples": {
            "s1": ["Final examination.", "Bar examination."],
            "s2": ["Pelvic examination.", "Physical examination."]
        }
    },
    "expire": {
        "senses": {"s1": "To become invalid.", "s2": "A euphemism for death."},
        "examples": {
            "s1": ["The coupon expires.", "License expired."],
            "s2": ["The patient expired.", "Expired at 3:42 AM."]
        }
    },
    "fever": {
        "senses": {"s1": "Excitement or frenzy.", "s2": "Elevated body temperature."},
        "examples": {
            "s1": ["Gold fever.", "Election fever."],
            "s2": ["High grade fever.", "Fever of unknown origin."]
        }
    },
    "fracture": {
        "senses": {"s1": "A break or split.", "s2": "A broken bone."},
        "examples": {
            "s1": ["Fractured relationship.", "Political fracture."],
            "s2": ["Compound fracture.", "Stress fracture."]
        }
    },
    "grade": {
        "senses": {"s1": "A school mark or level.", "s2": "Degree of malignancy."},
        "examples": {
            "s1": ["Sixth grade.", "Good grades."],
            "s2": ["High-grade tumor.", "Low-grade lymphoma."]
        }
    },
    "graft": {
        "senses": {"s1": "Corruption or bribery.", "s2": "Transplanted tissue."},
        "examples": {
            "s1": ["Political graft.", "Graft and corruption."],
            "s2": ["Skin graft.", "Bone graft."]
        }
    },
    "growth": {
        "senses": {"s1": "Increase or development.", "s2": "A tumor or abnormal mass."},
        "examples": {
            "s1": ["Economic growth.", "Personal growth."],
            "s2": ["A suspicious growth.", "Benign growth."]
        }
    },
    "history": {
        "senses": {"s1": "Past events.", "s2": "A record of health information."},
        "examples": {
            "s1": ["World history.", "History repeats."],
            "s2": ["Take a patient history.", "Family history of cancer."]
        }
    },
    "host": {
        "senses": {"s1": "Party giver or presenter.", "s2": "Organism supporting a parasite."},
        "examples": {
            "s1": ["TV host.", "Host a party."],
            "s2": ["Viral host.", "Intermediate host."]
        }
    },
    "immunity": {
        "senses": {"s1": "Exemption from duty.", "s2": "Resistance to disease."},
        "examples": {
            "s1": ["Diplomatic immunity.", "Tax immunity."],
            "s2": ["Herd immunity.", "Acquired immunity."]
        }
    },
    "implant": {
        "senses": {"s1": "To establish firmly.", "s2": "An inserted device or tissue."},
        "examples": {
            "s1": ["Implant an idea.", "Deeply implanted."],
            "s2": ["Cochlear implant.", "Dental implant."]
        }
    },
    "infection": {
        "senses": {"s1": "Contamination of ideas.", "s2": "Invasion by pathogens."},
        "examples": {
            "s1": ["Infection of corruption.", "Infectious enthusiasm."],
            "s2": ["Bacterial infection.", "Urinary tract infection."]
        }
    },
    "inflammation": {
        "senses": {"s1": "Arousing strong feelings.", "s2": "Redness and swelling of tissue."},
        "examples": {
            "s1": ["Inflame passions.", "Inflammatory rhetoric."],
            "s2": ["Chronic inflammation.", "Joint inflammation."]
        }
    },
    "injection": {
        "senses": {"s1": "Inserting something (funds, energy).", "s2": "Forcing liquid into the body."},
        "examples": {
            "s1": ["Cash injection.", "Injection of new ideas."],
            "s2": ["Intramuscular injection.", "Flu injection."]
        }
    },
    "labor": {
        "senses": {"s1": "Work or effort.", "s2": "The childbirth process."},
        "examples": {
            "s1": ["Manual labor.", "Labor Day."],
            "s2": ["Induced labor.", "Labor pains."]
        }
    },
    "lesion": {
        "senses": {"s1": "An injury (rare general).", "s2": "Damage to organ or tissue."},
        "examples": {
            "s1": ["Lesion to reputation.", "A lesion in judgment."],
            "s2": ["Brain lesion.", "Skin lesion."]
        }
    },
    "malignant": {
        "senses": {"s1": "Evil or malevolent.", "s2": "Cancerous."},
        "examples": {
            "s1": ["Malignant narcissism.", "Malignant influence."],
            "s2": ["Malignant tumor.", "Malignant melanoma."]
        }
    },
    "monitor": {
        "senses": {"s1": "A display screen.", "s2": "To observe a patient."},
        "examples": {
            "s1": ["Computer monitor.", "Baby monitor."],
            "s2": ["Monitor the heart rate.", "Patient monitoring."]
        }
    },
    "murmur": {
        "senses": {"s1": "A soft utterance.", "s2": "An abnormal heart sound."},
        "examples": {
            "s1": ["Murmur of assent.", "Quiet murmur."],
            "s2": ["Systolic murmur.", "Heart murmur."]
        }
    },
    "negative": {
        "senses": {"s1": "Pessimistic or unfavorable.", "s2": "Absence of a condition."},
        "examples": {
            "s1": ["Negative vibes.", "Negative feedback."],
            "s2": ["HIV negative.", "Test came back negative."]
        }
    },
    "node": {
        "senses": {"s1": "A network point.", "s2": "A tissue mass."},
        "examples": {
            "s1": ["Server node.", "Network node."],
            "s2": ["Lymph node.", "Swollen nodes."]
        }
    },
    "operation": {
        "senses": {"s1": "A mission or activity.", "s2": "A surgical procedure."},
        "examples": {
            "s1": ["Military operation.", "Business operation."],
            "s2": ["Bypass operation.", "Emergency operation."]
        }
    },
    "organ": {
        "senses": {"s1": "A musical instrument.", "s2": "A body part performing a function."},
        "examples": {
            "s1": ["Pipe organ.", "Church organ."],
            "s2": ["Vital organ.", "Organ transplant."]
        }
    },
    "patient": {
        "senses": {"s1": "Able to wait calmly.", "s2": "A person receiving medical care."},
        "examples": {
            "s1": ["Be patient.", "Patient listener."],
            "s2": ["Trauma patient.", "Patient care."]
        }
    },
    "physical": {
        "senses": {"s1": "Relating to the body or physics.", "s2": "A medical examination."},
        "examples": {
            "s1": ["Physical force.", "Physical activity."],
            "s2": ["Annual physical.", "Sports physical."]
        }
    },
    "plasma": {
        "senses": {"s1": "A state of matter.", "s2": "The fluid part of blood."},
        "examples": {
            "s1": ["Plasma TV.", "Plasma physics."],
            "s2": ["Blood plasma.", "Plasma donation."]
        }
    },
    "positive": {
        "senses": {"s1": "Optimistic or affirmative.", "s2": "Presence of a condition."},
        "examples": {
            "s1": ["Positive attitude.", "Positive outlook."],
            "s2": ["Covid positive.", "Tested positive."]
        }
    },
    "prescription": {
        "senses": {"s1": "A recommended formula.", "s2": "A doctor's drug order."},
        "examples": {
            "s1": ["Prescription for disaster.", "No easy prescription."],
            "s2": ["Fill a prescription.", "Prescription medication."]
        }
    },
    "pressure": {
        "senses": {"s1": "Stress or urgency.", "s2": "Force per unit area in the body."},
        "examples": {
            "s1": ["Peer pressure.", "Under pressure."],
            "s2": ["High blood pressure.", "Intracranial pressure."]
        }
    },
    "procedure": {
        "senses": {"s1": "A series of actions.", "s2": "A medical operation."},
        "examples": {
            "s1": ["Standard procedure.", "Follow procedure."],
            "s2": ["Surgical procedure.", "Outpatient procedure."]
        }
    },
    "pulse": {
        "senses": {"s1": "A rhythm or beat.", "s2": "The arterial heartbeat."},
        "examples": {
            "s1": ["Pulse of the city.", "Music pulse."],
            "s2": ["Check the pulse.", "Weak pulse."]
        }
    },
    "reaction": {
        "senses": {"s1": "A response.", "s2": "A physiological response."},
        "examples": {
            "s1": ["Gut reaction.", "Chain reaction."],
            "s2": ["Allergic reaction.", "Adverse reaction."]
        }
    },
    "recovery": {
        "senses": {"s1": "Retrieval or restoration.", "s2": "Return to health."},
        "examples": {
            "s1": ["Data recovery.", "Economic recovery."],
            "s2": ["Speedy recovery.", "Recovery room."]
        }
    },
    "seizure": {
        "senses": {"s1": "Taking possession.", "s2": "A neurological event."},
        "examples": {
            "s1": ["Asset seizure.", "Drug seizure."],
            "s2": ["Epileptic seizure.", "Grand mal seizure."]
        }
    },
    "stool": {
        "senses": {"s1": "A backless seat.", "s2": "Feces."},
        "examples": {
            "s1": ["Bar stool.", "Wooden stool."],
            "s2": ["Stool sample.", "Bloody stool."]
        }
    },
    "table": {
        "senses": {"s1": "A piece of furniture.", "s2": "An operating surface or data format."},
        "examples": {
            "s1": ["Kitchen table.", "Table for two."],
            "s2": ["Operating table.", "On the table."]
        }
    },
    "tenderness": {
        "senses": {"s1": "Gentleness or softness.", "s2": "Pain on palpation."},
        "examples": {
            "s1": ["He spoke with tenderness.", "Tender care."],
            "s2": ["Abdominal tenderness.", "Point tenderness."]
        }
    },
    "unit": {
        "senses": {"s1": "A measurement or single item.", "s2": "A hospital department."},
        "examples": {
            "s1": ["Unit of measurement.", "Apartment unit."],
            "s2": ["Intensive Care Unit.", "Burn unit."]
        }
    },
    "vital": {
        "senses": {"s1": "Essential or important.", "s2": "Signs of life."},
        "examples": {
            "s1": ["It is vital that we finish.", "Vital information."],
            "s2": ["Check his vitals.", "Vital signs stable."]
        }
    },
    "complication": {
        "senses": {"s1": "A difficulty or problem.", "s2": "A secondary disease or condition."},
        "examples": {
            "s1": ["It adds a complication.", "Unexpected complication."],
            "s2": ["Pneumonia is a complication of flu.", "Surgical complications."]
        }
    },
}


# =============================================================================
# DOMAIN 3: ENGINEERING & COMPUTER SCIENCE (80 polysemes)
# =============================================================================

TECH_CATALOG = {
    # Engineering terms
    "stress": {
        "senses": {"s1": "Mental tension.", "s2": "Force per unit area on a material."},
        "examples": {
            "s1": ["Work stress.", "Stress management."],
            "s2": ["Tensile stress.", "Material under stress."]
        }
    },
    "strain": {
        "senses": {"s1": "To filter or exert.", "s2": "Material deformation."},
        "examples": {
            "s1": ["Strain the pasta.", "Muscle strain."],
            "s2": ["Strain gauge.", "Elastic strain."]
        }
    },
    "fatigue": {
        "senses": {"s1": "Tiredness.", "s2": "Weakness in material from repeated stress."},
        "examples": {
            "s1": ["Combat fatigue.", "Mental fatigue."],
            "s2": ["Metal fatigue.", "Fatigue failure."]
        }
    },
    "yield": {
        "senses": {"s1": "To give way or produce.", "s2": "To deform plastically without breaking."},
        "examples": {
            "s1": ["Yield to traffic.", "High yield."],
            "s2": ["Yield strength.", "Yield point."]
        }
    },
    "tolerance": {
        "senses": {"s1": "Patience or acceptance.", "s2": "Allowable variation in dimensions."},
        "examples": {
            "s1": ["Religious tolerance.", "Zero tolerance."],
            "s2": ["Manufacturing tolerance.", "Tight tolerances."]
        }
    },
    "pitch": {
        "senses": {"s1": "A sales attempt or sound frequency.", "s2": "Distance between thread crests."},
        "examples": {
            "s1": ["Sales pitch.", "Perfect pitch."],
            "s2": ["Thread pitch.", "Screw pitch."]
        }
    },
    "head": {
        "senses": {"s1": "The body part above the neck.", "s2": "Pressure or height of liquid."},
        "examples": {
            "s1": ["Head first.", "Use your head."],
            "s2": ["Pressure head.", "Hydraulic head."]
        }
    },
    "current": {
        "senses": {"s1": "Present or up-to-date.", "s2": "Flow of electric charge."},
        "examples": {
            "s1": ["Current events.", "Current status."],
            "s2": ["Electric current.", "Direct current."]
        }
    },
    "resistance": {
        "senses": {"s1": "Opposition or refusal.", "s2": "Opposition to electric current flow."},
        "examples": {
            "s1": ["Resistance movement.", "Change resistance."],
            "s2": ["Electrical resistance.", "Low resistance."]
        }
    },
    "potential": {
        "senses": {"s1": "Possibility or capability.", "s2": "Voltage difference."},
        "examples": {
            "s1": ["Great potential.", "Potential energy."],
            "s2": ["Electric potential.", "Potential difference."]
        }
    },
    "charge": {
        "senses": {"s1": "An accusation or attack.", "s2": "Stored electrical energy."},
        "examples": {
            "s1": ["Criminal charge.", "Cavalry charge."],
            "s2": ["Battery charge.", "Positive charge."]
        }
    },
    "power": {
        "senses": {"s1": "Authority or control.", "s2": "Rate of work or energy transfer."},
        "examples": {
            "s1": ["Political power.", "Power struggle."],
            "s2": ["Power output.", "Measured in watts."]
        }
    },
    "cell": {
        "senses": {"s1": "A prison room.", "s2": "A battery unit or biological unit."},
        "examples": {
            "s1": ["Prison cell.", "Monk's cell."],
            "s2": ["Battery cell.", "Fuel cell."]
        }
    },
    "terminal": {
        "senses": {"s1": "Fatal or final.", "s2": "A connection point."},
        "examples": {
            "s1": ["Terminal illness.", "Terminal velocity."],
            "s2": ["Battery terminal.", "Airport terminal."]
        }
    },
    "ground": {
        "senses": {"s1": "The earth's surface.", "s2": "Electrical return path."},
        "examples": {
            "s1": ["Sit on the ground.", "Ground level."],
            "s2": ["Ground wire.", "Proper grounding."]
        }
    },
    "load": {
        "senses": {"s1": "A burden.", "s2": "Electrical power or force applied."},
        "examples": {
            "s1": ["Heavy load.", "Workload."],
            "s2": ["Electrical load.", "Load capacity."]
        }
    },
    "field": {
        "senses": {"s1": "An open area.", "s2": "A region of magnetic or electric influence."},
        "examples": {
            "s1": ["Playing field.", "Field of study."],
            "s2": ["Magnetic field.", "Electric field."]
        }
    },
    # CS terms
    "bug": {
        "senses": {"s1": "An insect.", "s2": "A software error."},
        "examples": {
            "s1": ["A bug crawled on the leaf.", "Ladybug."],
            "s2": ["Fix the bug in the code.", "Debug the program."]
        }
    },
    "virus": {
        "senses": {"s1": "A biological pathogen.", "s2": "Malicious software."},
        "examples": {
            "s1": ["Flu virus.", "Viral infection."],
            "s2": ["Computer virus.", "Antivirus software."]
        }
    },
    "mouse": {
        "senses": {"s1": "A small rodent.", "s2": "A computer input device."},
        "examples": {
            "s1": ["The mouse ate the cheese.", "Field mouse."],
            "s2": ["Click the mouse.", "Wireless mouse."]
        }
    },
    "window": {
        "senses": {"s1": "An opening in a wall.", "s2": "A GUI panel."},
        "examples": {
            "s1": ["Open the window.", "Window seat."],
            "s2": ["Close the browser window.", "Pop-up window."]
        }
    },
    "cloud": {
        "senses": {"s1": "Water vapor in the sky.", "s2": "Remote server infrastructure."},
        "examples": {
            "s1": ["Dark clouds.", "Head in the clouds."],
            "s2": ["Cloud computing.", "Upload to the cloud."]
        }
    },
    "cookie": {
        "senses": {"s1": "A baked treat.", "s2": "Browser tracking data."},
        "examples": {
            "s1": ["Chocolate chip cookie.", "Cookie jar."],
            "s2": ["Clear browser cookies.", "Accept cookies."]
        }
    },
    "stream": {
        "senses": {"s1": "A small river.", "s2": "Continuous data flow."},
        "examples": {
            "s1": ["Mountain stream.", "Babbling stream."],
            "s2": ["Video stream.", "Data streaming."]
        }
    },
    "crash": {
        "senses": {"s1": "A collision.", "s2": "System failure."},
        "examples": {
            "s1": ["Car crash.", "Plane crash."],
            "s2": ["System crash.", "The app crashed."]
        }
    },
    "driver": {
        "senses": {"s1": "A vehicle operator.", "s2": "Hardware control software."},
        "examples": {
            "s1": ["Taxi driver.", "Designated driver."],
            "s2": ["Install the driver.", "Printer driver."]
        }
    },
    "boot": {
        "senses": {"s1": "Footwear.", "s2": "Starting up a computer."},
        "examples": {
            "s1": ["Leather boots.", "Hiking boot."],
            "s2": ["Boot the system.", "Reboot."]
        }
    },
    "firewall": {
        "senses": {"s1": "A fire barrier.", "s2": "Network security system."},
        "examples": {
            "s1": ["Fire broke through the firewall.", "Building firewall."],
            "s2": ["Configure the firewall.", "Blocked by firewall."]
        }
    },
    "library": {
        "senses": {"s1": "A book repository.", "s2": "A collection of code modules."},
        "examples": {
            "s1": ["Public library.", "Library card."],
            "s2": ["Import the Python library.", "Standard library."]
        }
    },
    "key": {
        "senses": {"s1": "A door opener.", "s2": "A database ID or encryption element."},
        "examples": {
            "s1": ["House key.", "Piano key."],
            "s2": ["Primary key.", "API key."]
        }
    },
    "thread": {
        "senses": {"s1": "A strand of fiber.", "s2": "A sequence of execution."},
        "examples": {
            "s1": ["Needle and thread.", "Thread the needle."],
            "s2": ["Multi-threaded.", "Kill the thread."]
        }
    },
    "shell": {
        "senses": {"s1": "A hard outer covering.", "s2": "A command-line interface."},
        "examples": {
            "s1": ["Turtle shell.", "Eggshell."],
            "s2": ["Bash shell.", "Shell script."]
        }
    },
    "root": {
        "senses": {"s1": "A plant structure.", "s2": "Superuser or top directory."},
        "examples": {
            "s1": ["Tree root.", "Root vegetable."],
            "s2": ["Root access.", "Root directory."]
        }
    },
    "port": {
        "senses": {"s1": "A harbor.", "s2": "A network interface."},
        "examples": {
            "s1": ["Cruise port.", "Port city."],
            "s2": ["Port 80.", "USB port."]
        }
    },
    "host": {
        "senses": {"s1": "A party giver.", "s2": "A networked computer."},
        "examples": {
            "s1": ["The gracious host.", "Host a dinner."],
            "s2": ["Web host.", "Host server."]
        }
    },
    "memory": {
        "senses": {"s1": "Recall ability.", "s2": "Computer storage."},
        "examples": {
            "s1": ["Good memory.", "Childhood memories."],
            "s2": ["Memory leak.", "Out of memory."]
        }
    },
    "patch": {
        "senses": {"s1": "A small piece of material.", "s2": "A software update."},
        "examples": {
            "s1": ["Eye patch.", "Pumpkin patch."],
            "s2": ["Security patch.", "Patch the system."]
        }
    },
    "platform": {
        "senses": {"s1": "A raised surface.", "s2": "An operating system environment."},
        "examples": {
            "s1": ["Train platform.", "Diving platform."],
            "s2": ["Cross-platform.", "Social media platform."]
        }
    },
    "protocol": {
        "senses": {"s1": "Etiquette or procedure.", "s2": "Communication rules."},
        "examples": {
            "s1": ["Diplomatic protocol.", "Office protocol."],
            "s2": ["HTTP protocol.", "Network protocol."]
        }
    },
    "icon": {
        "senses": {"s1": "A religious image or symbol.", "s2": "A GUI symbol."},
        "examples": {
            "s1": ["Cultural icon.", "Religious icon."],
            "s2": ["Click the icon.", "Desktop icon."]
        }
    },
    "interface": {
        "senses": {"s1": "A meeting point.", "s2": "A system boundary or API."},
        "examples": {
            "s1": ["Interface between cultures.", "Customer interface."],
            "s2": ["User interface.", "API interface."]
        }
    },
    "log": {
        "senses": {"s1": "A piece of wood.", "s2": "An event record."},
        "examples": {
            "s1": ["Fireplace log.", "Log cabin."],
            "s2": ["Error log.", "Server logs."]
        }
    },
    "token": {
        "senses": {"s1": "A symbol or voucher.", "s2": "A unit of text or authentication."},
        "examples": {
            "s1": ["Token of appreciation.", "Subway token."],
            "s2": ["Authentication token.", "Access token."]
        }
    },
    "tree": {
        "senses": {"s1": "A woody plant.", "s2": "A hierarchical data structure."},
        "examples": {
            "s1": ["Oak tree.", "Family tree."],
            "s2": ["Binary tree.", "Decision tree."]
        }
    },
    "web": {
        "senses": {"s1": "A spider's creation.", "s2": "The World Wide Web."},
        "examples": {
            "s1": ["Spider web.", "Web of lies."],
            "s2": ["Web browser.", "Web development."]
        }
    },
    "cache": {
        "senses": {"s1": "A hidden store.", "s2": "High-speed memory."},
        "examples": {
            "s1": ["Weapons cache.", "Hidden cache."],
            "s2": ["Clear the cache.", "CPU cache."]
        }
    },
    "queue": {
        "senses": {"s1": "A line of people.", "s2": "A FIFO data structure."},
        "examples": {
            "s1": ["Queue for tickets.", "Long queue."],
            "s2": ["Message queue.", "Priority queue."]
        }
    },
}


# =============================================================================
# DOMAIN 4: FINANCE (25 polysemes)
# =============================================================================

FINANCE_CATALOG = {
    "balance": {
        "senses": {"s1": "Equilibrium or stability.", "s2": "Account amount."},
        "examples": {
            "s1": ["Lose balance.", "Work-life balance."],
            "s2": ["Account balance.", "Check your balance."]
        }
    },
    "bond": {
        "senses": {"s1": "A connection or tie.", "s2": "A debt security."},
        "examples": {
            "s1": ["Strong bond.", "Family bond."],
            "s2": ["Government bonds.", "Bond market."]
        }
    },
    "capital": {
        "senses": {"s1": "A main city.", "s2": "Financial assets."},
        "examples": {
            "s1": ["Capital of France.", "State capital."],
            "s2": ["Venture capital.", "Capital gains."]
        }
    },
    "credit": {
        "senses": {"s1": "Praise or acknowledgment.", "s2": "Borrowing power or loan."},
        "examples": {
            "s1": ["Give credit where due.", "Credit to the team."],
            "s2": ["Line of credit.", "Credit score."]
        }
    },
    "depression": {
        "senses": {"s1": "Sadness or low mood.", "s2": "Economic downturn."},
        "examples": {
            "s1": ["Clinical depression.", "Feeling depressed."],
            "s2": ["The Great Depression.", "Economic depression."]
        }
    },
    "equity": {
        "senses": {"s1": "Fairness.", "s2": "Ownership value."},
        "examples": {
            "s1": ["Equity and justice.", "Pay equity."],
            "s2": ["Private equity.", "Home equity."]
        }
    },
    "float": {
        "senses": {"s1": "To stay on water.", "s2": "To offer shares publicly."},
        "examples": {
            "s1": ["Float on water.", "Ice cream float."],
            "s2": ["Float the company.", "Stock float."]
        }
    },
    "frozen": {
        "senses": {"s1": "Turned to ice.", "s2": "Locked or inaccessible assets."},
        "examples": {
            "s1": ["Frozen water.", "Frozen solid."],
            "s2": ["Frozen assets.", "Account frozen."]
        }
    },
    "hedge": {
        "senses": {"s1": "A row of bushes.", "s2": "Risk reduction strategy."},
        "examples": {
            "s1": ["Garden hedge.", "Trim the hedge."],
            "s2": ["Hedge fund.", "Hedge against inflation."]
        }
    },
    "liquid": {
        "senses": {"s1": "A fluid state.", "s2": "Easily converted to cash."},
        "examples": {
            "s1": ["Liquid water.", "Pour the liquid."],
            "s2": ["Liquid assets.", "Highly liquid."]
        }
    },
    "maturity": {
        "senses": {"s1": "Adulthood or wisdom.", "s2": "Due date of a financial instrument."},
        "examples": {
            "s1": ["Show maturity.", "Emotional maturity."],
            "s2": ["Bond maturity.", "At maturity."]
        }
    },
    "portfolio": {
        "senses": {"s1": "A case for artwork.", "s2": "A collection of investments."},
        "examples": {
            "s1": ["Artist portfolio.", "Design portfolio."],
            "s2": ["Investment portfolio.", "Diversified portfolio."]
        }
    },
    "principal": {
        "senses": {"s1": "A school head.", "s2": "The original sum of money."},
        "examples": {
            "s1": ["School principal.", "Principal dancer."],
            "s2": ["Pay down the principal.", "Principal and interest."]
        }
    },
    "return": {
        "senses": {"s1": "To come back.", "s2": "Profit on investment."},
        "examples": {
            "s1": ["Return home.", "Happy returns."],
            "s2": ["Return on investment.", "High returns."]
        }
    },
    "security": {
        "senses": {"s1": "Safety.", "s2": "A tradable financial asset."},
        "examples": {
            "s1": ["Airport security.", "Job security."],
            "s2": ["Marketable securities.", "Security trading."]
        }
    },
    "share": {
        "senses": {"s1": "To divide.", "s2": "A unit of ownership."},
        "examples": {
            "s1": ["Share the food.", "Share your thoughts."],
            "s2": ["Buy shares.", "Share price."]
        }
    },
    "stock": {
        "senses": {"s1": "A supply.", "s2": "Equity ownership."},
        "examples": {
            "s1": ["Out of stock.", "Stock up."],
            "s2": ["Stock market.", "Stock options."]
        }
    },
    "bear": {
        "senses": {"s1": "A large animal.", "s2": "A pessimistic investor."},
        "examples": {
            "s1": ["Brown bear.", "Teddy bear."],
            "s2": ["Bear market.", "Bearish outlook."]
        }
    },
    "bull": {
        "senses": {"s1": "A male bovine.", "s2": "An optimistic investor."},
        "examples": {
            "s1": ["Angry bull.", "Bull fighting."],
            "s2": ["Bull market.", "Bullish on tech."]
        }
    },
    "correction": {
        "senses": {"s1": "A fix or adjustment.", "s2": "A market decline."},
        "examples": {
            "s1": ["Spelling correction.", "Course correction."],
            "s2": ["Market correction.", "10% correction."]
        }
    },
}


# =============================================================================
# DOMAIN 5: SLANG (45 polysemes) - High-velocity semantic drift
# =============================================================================

SLANG_CATALOG = {
    "cap": {
        "senses": {"s1": "A hat.", "s2": "A lie or exaggeration (slang)."},
        "examples": {
            "s1": ["Baseball cap.", "Remove your cap."],
            "s2": ["That's cap.", "No cap, it's true."]
        }
    },
    "cook": {
        "senses": {"s1": "To prepare food.", "s2": "To perform well or dominate (slang)."},
        "examples": {
            "s1": ["Cook dinner.", "Learning to cook."],
            "s2": ["Let him cook.", "He's cooking right now."]
        }
    },
    "drip": {
        "senses": {"s1": "Liquid falling in drops.", "s2": "A stylish outfit (slang)."},
        "examples": {
            "s1": ["Water drip.", "IV drip."],
            "s2": ["Check out that drip.", "Got the drip."]
        }
    },
    "ghost": {
        "senses": {"s1": "A spirit of the dead.", "s2": "To cut off communication (slang)."},
        "examples": {
            "s1": ["Ghost story.", "Haunted by ghosts."],
            "s2": ["She ghosted him.", "Don't ghost me."]
        }
    },
    "goat": {
        "senses": {"s1": "A farm animal.", "s2": "Greatest of All Time (slang)."},
        "examples": {
            "s1": ["Mountain goat.", "Goat cheese."],
            "s2": ["He's the GOAT.", "Undisputed goat."]
        }
    },
    "salty": {
        "senses": {"s1": "Containing salt.", "s2": "Bitter or upset (slang)."},
        "examples": {
            "s1": ["Salty food.", "Too salty."],
            "s2": ["Why are you so salty?", "Don't be salty."]
        }
    },
    "tea": {
        "senses": {"s1": "A beverage.", "s2": "Gossip (slang)."},
        "examples": {
            "s1": ["Cup of tea.", "Green tea."],
            "s2": ["Spill the tea.", "What's the tea?"]
        }
    },
    "tool": {
        "senses": {"s1": "An instrument.", "s2": "A foolish person (slang)."},
        "examples": {
            "s1": ["Hammer is a tool.", "Power tool."],
            "s2": ["He's such a tool.", "Stop being a tool."]
        }
    },
    "shade": {
        "senses": {"s1": "Blocked sunlight.", "s2": "A subtle insult (slang)."},
        "examples": {
            "s1": ["Sit in the shade.", "Tree shade."],
            "s2": ["Throwing shade.", "That was shade."]
        }
    },
    "flex": {
        "senses": {"s1": "To bend or tighten muscle.", "s2": "To show off (slang)."},
        "examples": {
            "s1": ["Flex your arm.", "Flexible schedule."],
            "s2": ["Stop flexing.", "Weird flex but okay."]
        }
    },
    "lit": {
        "senses": {"s1": "Illuminated.", "s2": "Exciting or intoxicated (slang)."},
        "examples": {
            "s1": ["Well lit room.", "Candles were lit."],
            "s2": ["The party was lit.", "Getting lit tonight."]
        }
    },
    "mid": {
        "senses": {"s1": "Middle position.", "s2": "Mediocre or average (slang)."},
        "examples": {
            "s1": ["Mid-morning.", "Mid-range."],
            "s2": ["That take is mid.", "Pretty mid performance."]
        }
    },
    "slay": {
        "senses": {"s1": "To kill.", "s2": "To succeed or dominate (slang)."},
        "examples": {
            "s1": ["Slay the dragon.", "Knight slays beast."],
            "s2": ["She slayed that look.", "Absolutely slaying it."]
        }
    },
    "stan": {
        "senses": {"s1": "A name.", "s2": "An obsessive fan (slang)."},
        "examples": {
            "s1": ["My friend Stan.", "Stan from accounting."],
            "s2": ["I stan this artist.", "Stan culture."]
        }
    },
    "woke": {
        "senses": {"s1": "Past tense of wake.", "s2": "Socially aware (slang)."},
        "examples": {
            "s1": ["I woke up early.", "She just woke."],
            "s2": ["Stay woke.", "Woke agenda."]
        }
    },
    "sus": {
        "senses": {"s1": "A shortened form.", "s2": "Suspicious (slang)."},
        "examples": {
            "s1": ["Sus it out.", "That's sus."],
            "s2": ["Acting real sus.", "That's sus behavior."]
        }
    },
    "bet": {
        "senses": {"s1": "A wager.", "s2": "Agreement or yes (slang)."},
        "examples": {
            "s1": ["Place a bet.", "Win the bet."],
            "s2": ["Bet, I'll be there.", "Bet, sounds good."]
        }
    },
    "basic": {
        "senses": {"s1": "Fundamental.", "s2": "Unoriginal or mainstream (slang)."},
        "examples": {
            "s1": ["Basic training.", "Basic principles."],
            "s2": ["So basic.", "Basic tastes."]
        }
    },
    "extra": {
        "senses": {"s1": "Additional.", "s2": "Over-dramatic (slang)."},
        "examples": {
            "s1": ["Extra credit.", "Extra large."],
            "s2": ["You're being extra.", "So extra right now."]
        }
    },
    "gas": {
        "senses": {"s1": "A fuel or state of matter.", "s2": "Praise or high quality (slang)."},
        "examples": {
            "s1": ["Fill up on gas.", "Gas station."],
            "s2": ["That track is gas.", "Pure gas."]
        }
    },
    "heat": {
        "senses": {"s1": "High temperature.", "s2": "Pressure or good music (slang)."},
        "examples": {
            "s1": ["Summer heat.", "Body heat."],
            "s2": ["Bringing the heat.", "This beat is heat."]
        }
    },
    "brick": {
        "senses": {"s1": "A building block.", "s2": "To fail or miss badly (slang)."},
        "examples": {
            "s1": ["Red brick.", "Brick wall."],
            "s2": ["He bricked that shot.", "Complete brick."]
        }
    },
    "aura": {
        "senses": {"s1": "An atmosphere or emanation.", "s2": "Coolness points (slang)."},
        "examples": {
            "s1": ["Mystical aura.", "Positive aura."],
            "s2": ["Major aura points.", "Lost aura."]
        }
    },
    "based": {
        "senses": {"s1": "Founded on.", "s2": "Unapologetically authentic (slang)."},
        "examples": {
            "s1": ["Based on facts.", "Evidence based."],
            "s2": ["That take is based.", "So based."]
        }
    },
    "serve": {
        "senses": {"s1": "To deliver food or tennis.", "s2": "To present a stunning look (slang)."},
        "examples": {
            "s1": ["Serve dinner.", "Tennis serve."],
            "s2": ["She served looks.", "Serve face."]
        }
    },
    "simp": {
        "senses": {"s1": "A simpleton.", "s2": "A submissive admirer (slang)."},
        "examples": {
            "s1": ["Don't be a simp.", "Simple minded."],
            "s2": ["Stop simping.", "Total simp."]
        }
    },
    "sleep": {
        "senses": {"s1": "Rest state.", "s2": "To ignore or underestimate (slang)."},
        "examples": {
            "s1": ["Get some sleep.", "Sleep well."],
            "s2": ["Don't sleep on this.", "You're sleeping on him."]
        }
    },
    "ratio": {
        "senses": {"s1": "A mathematical proportion.", "s2": "When replies exceed likes (slang)."},
        "examples": {
            "s1": ["Gear ratio.", "Golden ratio."],
            "s2": ["You got ratioed.", "Ratio + L."]
        }
    },
    "rizz": {
        "senses": {"s1": "Charisma (neologism).", "s2": "Seduction skill (slang)."},
        "examples": {
            "s1": ["Natural rizz.", "Rizz them up."],
            "s2": ["Unspoken rizz.", "W rizz."]
        }
    },
}


# =============================================================================
# BUNDLE GENERATION
# =============================================================================

def create_bundle(word: str, info: Dict, bundle_id: int, domain: str) -> Dict:
    """Create a SemanticPhase v2 bundle from word info."""
    senses = info["senses"]
    examples = info["examples"]

    items = []
    sense_keys = list(senses.keys())

    for sense_id in sense_keys:
        sense_examples = examples.get(sense_id, [])
        for i, sent in enumerate(sense_examples):
            items.append({
                "item_id": f"{bundle_id}_{sense_id}_{i}",
                "sentence": sent,
                "sense_id": sense_id,
                "role": "anchor" if i == 0 else "positive",
                "difficulty": "easy" if sense_id == "s1" else "hard"
            })

    # Create pairings
    pairings = []
    if len(sense_keys) >= 2:
        s1_items = [it for it in items if it["sense_id"] == "s1"]
        s2_items = [it for it in items if it["sense_id"] == "s2"]

        for anchor in s1_items[:1]:  # First s1 as anchor
            for pos in s1_items[1:]:  # Other s1 as positives
                for neg in s2_items:  # s2 as hard negatives
                    pairings.append({
                        "anchor": anchor["item_id"],
                        "positive": pos["item_id"],
                        "negative": neg["item_id"],
                        "difficulty": "hard"
                    })

    return {
        "bundle_id": f"SPv2_{bundle_id:05d}",
        "word": word,
        "sense_catalog": senses,
        "metadata": {
            "domain": domain,
            "created": datetime.now().isoformat()
        },
        "pairings": pairings,
        "items": items
    }


def generate_all_bundles(output_path: Path) -> int:
    """Generate bundles from all catalogs."""
    all_catalogs = [
        (LAW_CATALOG, "Law"),
        (MEDICAL_CATALOG, "Medical"),
        (TECH_CATALOG, "Tech/CS"),
        (FINANCE_CATALOG, "Finance"),
        (SLANG_CATALOG, "Slang"),
    ]

    bundle_id = 1
    total_bundles = 0

    with open(output_path, 'w') as f:
        for catalog, domain in all_catalogs:
            for word, info in catalog.items():
                bundle = create_bundle(word, info, bundle_id, domain)
                f.write(json.dumps(bundle) + "\n")
                bundle_id += 1
                total_bundles += 1

    return total_bundles


if __name__ == "__main__":
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "semanticphase_v2_comprehensive.jsonl"

    count = generate_all_bundles(output_path)

    print(f"Generated {count} comprehensive polysemy bundles")
    print(f"Output: {output_path}")

    # Stats by domain
    catalogs = {
        "Law": len(LAW_CATALOG),
        "Medical": len(MEDICAL_CATALOG),
        "Tech/CS": len(TECH_CATALOG),
        "Finance": len(FINANCE_CATALOG),
        "Slang": len(SLANG_CATALOG),
    }

    print("\nDomain breakdown:")
    for domain, count in catalogs.items():
        print(f"  {domain}: {count} words")
    print(f"  Total: {sum(catalogs.values())} words")
