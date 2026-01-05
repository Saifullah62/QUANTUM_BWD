#!/usr/bin/env python3
"""
SemanticPhase v2 Expanded Bundle Generator
===========================================

Generates high-quality polysemy bundles from the 250-word lexicographical catalog.
Covers 5 domains: Law/Finance, Science/Medical, Technology, Slang, General/Academic.

Schema: bundle_id, word, sense_catalog, metadata, pairings, items
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# =============================================================================
# DOMAIN 1: LAW & FINANCE (50 words)
# =============================================================================

LAW_FINANCE_CATALOG = {
    "consideration": {
        "senses": {
            "s1": "Careful thought over time.",
            "s2": "Value/payment given in a contract (legal)."
        },
        "domains": ["General", "Legal"],
        "examples": {
            "s1": ["After much consideration, he declined the offer.", "The matter deserves careful consideration."],
            "s2": ["The contract failed for lack of consideration.", "Consideration is required to make a binding agreement."]
        }
    },
    "stay": {
        "senses": {
            "s1": "To remain in a place.",
            "s2": "An order to suspend legal proceedings."
        },
        "domains": ["General", "Legal"],
        "examples": {
            "s1": ["I will stay here until you return.", "She decided to stay home."],
            "s2": ["Counsel requested a stay of execution.", "The judge granted a stay pending appeal."]
        }
    },
    "action": {
        "senses": {
            "s1": "The process of doing something.",
            "s2": "A lawsuit or legal proceeding."
        },
        "domains": ["General", "Legal"],
        "examples": {
            "s1": ["The action sequence was exciting.", "We need to take immediate action."],
            "s2": ["They filed an action against the company.", "The action was dismissed for lack of standing."]
        }
    },
    "instrument": {
        "senses": {
            "s1": "A tool or device for delicate work.",
            "s2": "A formal legal document (deed, will, contract)."
        },
        "domains": ["General", "Legal"],
        "examples": {
            "s1": ["The surgeon selected the correct instrument.", "A musical instrument requires practice."],
            "s2": ["The instrument was executed before witnesses.", "A negotiable instrument must be in writing."]
        }
    },
    "security": {
        "senses": {
            "s1": "Safety from danger or threat.",
            "s2": "A tradable financial asset (stock, bond)."
        },
        "domains": ["General", "Finance"],
        "examples": {
            "s1": ["The security of the building was improved.", "National security is a priority."],
            "s2": ["Securities prices fluctuate daily.", "She invested in government securities."]
        }
    },
    "equity": {
        "senses": {
            "s1": "Fairness and justice.",
            "s2": "Ownership value in an asset after debts."
        },
        "domains": ["Ethics", "Finance"],
        "examples": {
            "s1": ["The decision was based on equity.", "Courts of equity developed separate remedies."],
            "s2": ["Home equity has increased significantly.", "Private equity firms invest in companies."]
        }
    },
    "option": {
        "senses": {
            "s1": "A choice or alternative.",
            "s2": "A contractual right to buy/sell at a set price."
        },
        "domains": ["General", "Finance"],
        "examples": {
            "s1": ["You have several options available.", "What are my options here?"],
            "s2": ["Stock options are part of executive compensation.", "The call option expires in March."]
        }
    },
    "return": {
        "senses": {
            "s1": "Coming or going back to a place.",
            "s2": "Profit or income from an investment."
        },
        "domains": ["General", "Finance"],
        "examples": {
            "s1": ["She will return next week.", "Return to the starting point."],
            "s2": ["The return on investment was 12%.", "Investors expect reasonable returns."]
        }
    },
    "bench": {
        "senses": {
            "s1": "A long seat for multiple people.",
            "s2": "The judge or judiciary collectively."
        },
        "domains": ["Furniture", "Legal"],
        "examples": {
            "s1": ["They sat on a park bench.", "The bench was made of oak."],
            "s2": ["The bench ruled against the motion.", "He was elevated to the bench in 2015."]
        }
    },
    "trust": {
        "senses": {
            "s1": "Firm belief in reliability or truth.",
            "s2": "A legal arrangement for managing assets."
        },
        "domains": ["Personal", "Legal"],
        "examples": {
            "s1": ["Trust is essential in relationships.", "I trust your judgment completely."],
            "s2": ["The trust was established for the children.", "Assets were transferred to the family trust."]
        }
    },
    "will": {
        "senses": {
            "s1": "Determination or desire to do something.",
            "s2": "A testamentary document for asset distribution."
        },
        "domains": ["Personal", "Legal"],
        "examples": {
            "s1": ["She has the will to succeed.", "Where there's a will, there's a way."],
            "s2": ["The will was read after the funeral.", "He left everything to charity in his will."]
        }
    },
    "deed": {
        "senses": {
            "s1": "An action or act performed.",
            "s2": "A legal document transferring property ownership."
        },
        "domains": ["General", "Legal"],
        "examples": {
            "s1": ["Good deeds are their own reward.", "The deed was done before anyone could stop him."],
            "s2": ["The deed was recorded at the county office.", "Sign the deed to transfer the property."]
        }
    },
    "brief": {
        "senses": {
            "s1": "Short in duration or length.",
            "s2": "A written legal argument submitted to court."
        },
        "domains": ["General", "Legal"],
        "examples": {
            "s1": ["Keep your remarks brief.", "It was a brief meeting."],
            "s2": ["The attorney filed a brief with the court.", "The amicus brief supported the plaintiff."]
        }
    },
    "motion": {
        "senses": {
            "s1": "The act of moving.",
            "s2": "A formal proposal to a court or assembly."
        },
        "domains": ["Physical", "Legal"],
        "examples": {
            "s1": ["The motion of the waves was hypnotic.", "Set the ball in motion."],
            "s2": ["Defense filed a motion to dismiss.", "The motion was seconded and carried."]
        }
    },
    "hearing": {
        "senses": {
            "s1": "The faculty of perceiving sound.",
            "s2": "A court session for legal arguments."
        },
        "domains": ["Medical", "Legal"],
        "examples": {
            "s1": ["His hearing has declined with age.", "Within hearing distance."],
            "s2": ["The hearing is scheduled for Monday.", "A preliminary hearing was held."]
        }
    },
    "serve": {
        "senses": {
            "s1": "To provide food or assistance.",
            "s2": "To deliver legal documents officially."
        },
        "domains": ["Service", "Legal"],
        "examples": {
            "s1": ["Serve the guests first.", "He served in the military."],
            "s2": ["The summons was served yesterday.", "Service of process was completed."]
        }
    },
    "damages": {
        "senses": {
            "s1": "Physical harm or destruction (rare usage).",
            "s2": "Monetary compensation awarded by court."
        },
        "domains": ["General", "Legal"],
        "examples": {
            "s1": ["The storm caused extensive damages.", "Assess the damages to the vehicle."],
            "s2": ["The jury awarded $2 million in damages.", "Punitive damages were sought."]
        }
    },
    "relief": {
        "senses": {
            "s1": "Alleviation of pain or distress.",
            "s2": "Court-ordered remedy or redress."
        },
        "domains": ["Medical", "Legal"],
        "examples": {
            "s1": ["The medicine brought relief.", "What a relief that's over."],
            "s2": ["The plaintiff sought injunctive relief.", "Relief was granted by the court."]
        }
    },
    "dock": {
        "senses": {
            "s1": "A structure for mooring ships.",
            "s2": "The enclosure where a prisoner stands in court."
        },
        "domains": ["Maritime", "Legal"],
        "examples": {
            "s1": ["The boat is tied to the dock.", "We walked along the dock."],
            "s2": ["The defendant stood in the dock.", "She was brought to the dock for sentencing."]
        }
    },
    "liquid": {
        "senses": {
            "s1": "A substance that flows freely (water, oil).",
            "s2": "Easily convertible to cash (financial)."
        },
        "domains": ["Chemistry", "Finance"],
        "examples": {
            "s1": ["The liquid filled the container.", "Water is a liquid at room temperature."],
            "s2": ["Liquid assets can be sold quickly.", "The company maintains liquid reserves."]
        }
    },
    "float": {
        "senses": {
            "s1": "To stay on the surface of liquid.",
            "s2": "To issue shares to the public (IPO)."
        },
        "domains": ["Physical", "Finance"],
        "examples": {
            "s1": ["The boat floats on water.", "Ice floats because it's less dense."],
            "s2": ["The company will float shares next month.", "They floated the IPO at $50 per share."]
        }
    },
    "hedge": {
        "senses": {
            "s1": "A row of bushes forming a boundary.",
            "s2": "An investment to reduce risk."
        },
        "domains": ["Garden", "Finance"],
        "examples": {
            "s1": ["Trim the hedge regularly.", "The hedge separates the properties."],
            "s2": ["A hedge against inflation.", "Hedge funds use complex strategies."]
        }
    },
    "bull": {
        "senses": {
            "s1": "A male bovine animal.",
            "s2": "An optimistic investor expecting prices to rise."
        },
        "domains": ["Animals", "Finance"],
        "examples": {
            "s1": ["The bull charged across the field.", "Bulls are used in rodeos."],
            "s2": ["Bulls are buying heavily.", "We're in a bull market."]
        }
    },
    "bear": {
        "senses": {
            "s1": "A large furry mammal.",
            "s2": "A pessimistic investor expecting prices to fall."
        },
        "domains": ["Animals", "Finance"],
        "examples": {
            "s1": ["The bear caught a fish.", "Grizzly bears are dangerous."],
            "s2": ["Bears are selling their positions.", "The bear market lasted two years."]
        }
    },
    "maturity": {
        "senses": {
            "s1": "The state of being fully developed or adult.",
            "s2": "The date when a bond or loan becomes due."
        },
        "domains": ["Personal", "Finance"],
        "examples": {
            "s1": ["He showed great maturity for his age.", "Emotional maturity takes time."],
            "s2": ["The bond has a ten-year maturity.", "The loan reaches maturity in 2030."]
        }
    },
    "principal": {
        "senses": {
            "s1": "The head of a school.",
            "s2": "The original sum of money invested or loaned."
        },
        "domains": ["Education", "Finance"],
        "examples": {
            "s1": ["The principal addressed the students.", "Call the principal's office."],
            "s2": ["Pay down the principal first.", "Interest accrues on the principal."]
        }
    },
    "balance": {
        "senses": {
            "s1": "A state of equilibrium or stability.",
            "s2": "The amount remaining in an account."
        },
        "domains": ["Physical", "Finance"],
        "examples": {
            "s1": ["Maintain your balance carefully.", "Work-life balance is important."],
            "s2": ["Check your account balance.", "The balance was transferred."]
        }
    },
    "gross": {
        "senses": {
            "s1": "Disgusting or repulsive.",
            "s2": "Total amount before deductions."
        },
        "domains": ["General", "Finance"],
        "examples": {
            "s1": ["That smell is gross.", "Gross behavior will not be tolerated."],
            "s2": ["Gross income before taxes.", "The gross revenue was $5 million."]
        }
    },
    "portfolio": {
        "senses": {
            "s1": "A case for carrying artwork or documents.",
            "s2": "A collection of investments owned by a person."
        },
        "domains": ["Art", "Finance"],
        "examples": {
            "s1": ["Show your portfolio to the client.", "An artist's portfolio of work."],
            "s2": ["Diversify your investment portfolio.", "The portfolio includes stocks and bonds."]
        }
    },
    "speculation": {
        "senses": {
            "s1": "Forming a theory without firm evidence.",
            "s2": "High-risk investment for potential gain."
        },
        "domains": ["Thinking", "Finance"],
        "examples": {
            "s1": ["That's pure speculation.", "Speculation about the cause continues."],
            "s2": ["Real estate speculation is risky.", "Currency speculation drove the market."]
        }
    },
    "warrant": {
        "senses": {
            "s1": "To justify or call for something.",
            "s2": "Legal authorization for arrest or search."
        },
        "domains": ["General", "Legal"],
        "examples": {
            "s1": ["The situation warrants investigation.", "Does this warrant a response?"],
            "s2": ["The police obtained a search warrant.", "An arrest warrant was issued."]
        }
    },
    "charge": {
        "senses": {
            "s1": "To rush forward in attack.",
            "s2": "A formal accusation of a crime.",
            "s3": "A price asked for goods or services."
        },
        "domains": ["Military", "Legal", "Commerce"],
        "examples": {
            "s1": ["The cavalry began their charge.", "The bull charged at the matador."],
            "s2": ["The charges were dropped.", "He was charged with fraud."],
            "s3": ["What's the charge for delivery?", "There's no extra charge."]
        }
    },
    "conviction": {
        "senses": {
            "s1": "A firmly held belief.",
            "s2": "A formal declaration of guilt by a court."
        },
        "domains": ["Belief", "Legal"],
        "examples": {
            "s1": ["He spoke with conviction.", "Her convictions are unshakeable."],
            "s2": ["The conviction was overturned.", "Prior convictions affected sentencing."]
        }
    },
    "appeal": {
        "senses": {
            "s1": "Attractiveness or interest.",
            "s2": "A request for a higher court to review a decision."
        },
        "domains": ["Attraction", "Legal"],
        "examples": {
            "s1": ["The idea has broad appeal.", "What's the appeal of that show?"],
            "s2": ["They filed an appeal immediately.", "The appeal was denied."]
        }
    },
    "fine": {
        "senses": {
            "s1": "Of high quality; excellent.",
            "s2": "A monetary penalty for an offense."
        },
        "domains": ["Quality", "Legal"],
        "examples": {
            "s1": ["That's a fine painting.", "The weather is fine today."],
            "s2": ["He paid a $500 fine.", "Fines will be issued for violations."]
        }
    }
}

# =============================================================================
# DOMAIN 2: SCIENCE & MEDICINE (50 words)
# =============================================================================

SCIENCE_MEDICAL_CATALOG = {
    "culture": {
        "senses": {
            "s1": "The arts, customs, and social institutions of a society.",
            "s2": "The cultivation of bacteria or cells in a lab."
        },
        "domains": ["Sociology", "Biology"],
        "examples": {
            "s1": ["Western culture emphasizes individualism.", "Cultural differences should be respected."],
            "s2": ["The culture was positive for strep.", "Cell culture requires sterile conditions."]
        }
    },
    "positive": {
        "senses": {
            "s1": "Expressing affirmation or optimism.",
            "s2": "Indicating the presence of a condition (medical test)."
        },
        "domains": ["General", "Medical"],
        "examples": {
            "s1": ["Stay positive during hard times.", "A positive attitude helps."],
            "s2": ["The test came back positive.", "COVID-positive patients were isolated."]
        }
    },
    "negative": {
        "senses": {
            "s1": "Expressing denial or pessimism.",
            "s2": "Indicating the absence of a condition (medical test)."
        },
        "domains": ["General", "Medical"],
        "examples": {
            "s1": ["Don't be so negative.", "Negative thinking limits you."],
            "s2": ["The biopsy was negative.", "She tested negative for the virus."]
        }
    },
    "acute": {
        "senses": {
            "s1": "Sharp or keen in perception.",
            "s2": "Severe and sudden in onset (disease)."
        },
        "domains": ["Mental", "Medical"],
        "examples": {
            "s1": ["He has an acute sense of hearing.", "Her acute observations impressed everyone."],
            "s2": ["Acute appendicitis requires surgery.", "The patient presented with acute pain."]
        }
    },
    "benign": {
        "senses": {
            "s1": "Kind and gentle in nature.",
            "s2": "Not cancerous; not harmful."
        },
        "domains": ["Personality", "Medical"],
        "examples": {
            "s1": ["He has a benign temperament.", "A benign smile crossed her face."],
            "s2": ["The tumor was benign.", "Benign growths don't spread."]
        }
    },
    "labor": {
        "senses": {
            "s1": "Physical or mental work.",
            "s2": "The process of childbirth."
        },
        "domains": ["Work", "Medical"],
        "examples": {
            "s1": ["Manual labor is hard work.", "Labor costs are rising."],
            "s2": ["She went into labor at midnight.", "Labor lasted twelve hours."]
        }
    },
    "depression": {
        "senses": {
            "s1": "A mental health condition involving persistent sadness.",
            "s2": "An anatomical hollow or sunken area."
        },
        "domains": ["Psychology", "Anatomy"],
        "examples": {
            "s1": ["Depression affects millions of people.", "He was treated for clinical depression."],
            "s2": ["The depression in the skull was visible.", "A small depression marks the site."]
        }
    },
    "radical": {
        "senses": {
            "s1": "Favoring extreme political change.",
            "s2": "A medical treatment removing the root/source."
        },
        "domains": ["Politics", "Medical"],
        "examples": {
            "s1": ["Radical ideas were once considered dangerous.", "A radical overhaul of the system."],
            "s2": ["A radical mastectomy was performed.", "Radical surgery removes all affected tissue."]
        }
    },
    "administration": {
        "senses": {
            "s1": "Management of an organization.",
            "s2": "The giving of a medication or treatment."
        },
        "domains": ["Management", "Medical"],
        "examples": {
            "s1": ["The new administration took office.", "Hospital administration approved the plan."],
            "s2": ["Administration of the drug requires training.", "Oral administration is preferred."]
        }
    },
    "solution": {
        "senses": {
            "s1": "An answer to a problem.",
            "s2": "A homogeneous mixture of substances."
        },
        "domains": ["Problem-solving", "Chemistry"],
        "examples": {
            "s1": ["We found a solution to the issue.", "That's not a good solution."],
            "s2": ["Dissolve salt in water to make a solution.", "A saline solution was used."]
        }
    },
    "organic": {
        "senses": {
            "s1": "Produced without artificial chemicals.",
            "s2": "Relating to carbon-based compounds."
        },
        "domains": ["Agriculture", "Chemistry"],
        "examples": {
            "s1": ["Organic vegetables are pesticide-free.", "The store sells organic produce."],
            "s2": ["Organic chemistry studies carbon compounds.", "Organic molecules are essential for life."]
        }
    },
    "base": {
        "senses": {
            "s1": "The foundation or bottom support.",
            "s2": "A substance with pH greater than 7."
        },
        "domains": ["General", "Chemistry"],
        "examples": {
            "s1": ["The base of the statue is marble.", "Build from a solid base."],
            "s2": ["A base reacts with an acid.", "Sodium hydroxide is a strong base."]
        }
    },
    "period": {
        "senses": {
            "s1": "A length of time.",
            "s2": "Menstruation (medical).",
            "s3": "A row in the periodic table."
        },
        "domains": ["Time", "Medical", "Chemistry"],
        "examples": {
            "s1": ["A period of growth followed.", "The trial period ends Friday."],
            "s2": ["Her period started yesterday.", "Irregular periods require evaluation."],
            "s3": ["Elements in the same period have the same shells.", "Period 3 contains sodium to argon."]
        }
    },
    "element": {
        "senses": {
            "s1": "A component or essential part.",
            "s2": "A fundamental substance that cannot be broken down chemically."
        },
        "domains": ["General", "Chemistry"],
        "examples": {
            "s1": ["Time is an essential element.", "The key elements of the plan."],
            "s2": ["Oxygen is an element.", "The periodic table lists all elements."]
        }
    },
    "reduce": {
        "senses": {
            "s1": "To make smaller or less.",
            "s2": "To gain electrons in a chemical reaction."
        },
        "domains": ["General", "Chemistry"],
        "examples": {
            "s1": ["Reduce your expenses.", "The company reduced its workforce."],
            "s2": ["The ion was reduced to its elemental form.", "Reduction involves gaining electrons."]
        }
    },
    "concentrate": {
        "senses": {
            "s1": "To focus mental effort.",
            "s2": "To increase the strength of a solution."
        },
        "domains": ["Mental", "Chemistry"],
        "examples": {
            "s1": ["Concentrate on the task.", "I can't concentrate with noise."],
            "s2": ["Concentrate the solution by evaporation.", "The concentrated acid is dangerous."]
        }
    },
    "suspension": {
        "senses": {
            "s1": "Temporarily stopping something.",
            "s2": "Particles dispersed throughout a fluid."
        },
        "domains": ["General", "Chemistry"],
        "examples": {
            "s1": ["His suspension from school lasted a week.", "Suspension of the program was announced."],
            "s2": ["Shake the suspension before use.", "A colloidal suspension of particles."]
        }
    },
    "complex": {
        "senses": {
            "s1": "Consisting of many interconnected parts.",
            "s2": "A molecular entity with a central atom bound to ligands."
        },
        "domains": ["General", "Chemistry"],
        "examples": {
            "s1": ["The problem is complex.", "A complex system of regulations."],
            "s2": ["The iron complex was analyzed.", "Metal complexes have unique properties."]
        }
    },
    "force": {
        "senses": {
            "s1": "Coercion or violence.",
            "s2": "An influence that causes motion or change (physics)."
        },
        "domains": ["General", "Physics"],
        "examples": {
            "s1": ["They used force to enter.", "Force won't solve this."],
            "s2": ["Gravity is a fundamental force.", "Calculate the force applied."]
        }
    },
    "power": {
        "senses": {
            "s1": "Political control or authority.",
            "s2": "The rate of doing work (physics)."
        },
        "domains": ["Politics", "Physics"],
        "examples": {
            "s1": ["The king held absolute power.", "Power corrupts."],
            "s2": ["Power is measured in watts.", "The engine's power output is 200 hp."]
        }
    },
    "work": {
        "senses": {
            "s1": "Activity involving effort for a purpose.",
            "s2": "Force multiplied by distance (physics)."
        },
        "domains": ["Employment", "Physics"],
        "examples": {
            "s1": ["Work begins at 9 AM.", "Hard work pays off."],
            "s2": ["Work equals force times distance.", "No work is done without displacement."]
        }
    },
    "field": {
        "senses": {
            "s1": "An open area of land.",
            "s2": "A region where a force acts (physics)."
        },
        "domains": ["Geography", "Physics"],
        "examples": {
            "s1": ["The field was full of wildflowers.", "A football field."],
            "s2": ["The magnetic field surrounds the wire.", "Electric field lines point away from positive charges."]
        }
    },
    "resistance": {
        "senses": {
            "s1": "The act of opposing something.",
            "s2": "Opposition to the flow of electric current."
        },
        "domains": ["General", "Physics"],
        "examples": {
            "s1": ["There was resistance to the new policy.", "The resistance movement grew."],
            "s2": ["Resistance is measured in ohms.", "Low resistance allows current to flow."]
        }
    },
    "stress": {
        "senses": {
            "s1": "Mental or emotional tension.",
            "s2": "Force per unit area (physics/engineering)."
        },
        "domains": ["Psychology", "Physics"],
        "examples": {
            "s1": ["Work stress is affecting her health.", "Manage your stress levels."],
            "s2": ["The bridge can withstand high stress.", "Calculate the stress on the beam."]
        }
    },
    "strain": {
        "senses": {
            "s1": "Severe demand on physical or mental resources.",
            "s2": "Deformation of a material under stress."
        },
        "domains": ["General", "Physics"],
        "examples": {
            "s1": ["The strain of caregiving took its toll.", "Don't put strain on yourself."],
            "s2": ["The strain on the metal caused fracture.", "Strain is dimensionless."]
        }
    },
    "moment": {
        "senses": {
            "s1": "A very brief period of time.",
            "s2": "The turning effect of a force (physics)."
        },
        "domains": ["Time", "Physics"],
        "examples": {
            "s1": ["Wait a moment.", "That was a defining moment."],
            "s2": ["Calculate the moment about the pivot.", "The moment of the force is 10 Nm."]
        }
    },
    "focus": {
        "senses": {
            "s1": "Concentrated attention or effort.",
            "s2": "The point where light rays converge."
        },
        "domains": ["Mental", "Optics"],
        "examples": {
            "s1": ["Focus on what matters.", "Her focus was impressive."],
            "s2": ["Adjust the lens to bring the image into focus.", "The focal point is 10 cm away."]
        }
    },
    "cell": {
        "senses": {
            "s1": "A small room, especially in a prison.",
            "s2": "The basic structural unit of living organisms."
        },
        "domains": ["Prison", "Biology"],
        "examples": {
            "s1": ["The prisoner was kept in a cell.", "A monastery cell."],
            "s2": ["Red blood cells carry oxygen.", "Plant cells have walls."]
        }
    },
    "tissue": {
        "senses": {
            "s1": "Soft paper used for wiping.",
            "s2": "A group of cells with similar function."
        },
        "domains": ["Household", "Biology"],
        "examples": {
            "s1": ["Pass me a tissue.", "A box of tissues."],
            "s2": ["Muscle tissue contracts.", "Tissue samples were analyzed."]
        }
    },
    "organ": {
        "senses": {
            "s1": "A musical instrument with pipes.",
            "s2": "A body part with a specific function."
        },
        "domains": ["Music", "Biology"],
        "examples": {
            "s1": ["The organ played at the wedding.", "A pipe organ."],
            "s2": ["The liver is a vital organ.", "Organ donation saves lives."]
        }
    },
    "host": {
        "senses": {
            "s1": "A person who receives guests.",
            "s2": "An organism that harbors a parasite."
        },
        "domains": ["Social", "Biology"],
        "examples": {
            "s1": ["The host welcomed the guests.", "She was a gracious host."],
            "s2": ["The parasite requires a host to survive.", "Humans are the primary host."]
        }
    },
    "vector": {
        "senses": {
            "s1": "A quantity with magnitude and direction.",
            "s2": "An organism that transmits disease."
        },
        "domains": ["Mathematics", "Medical"],
        "examples": {
            "s1": ["Calculate the vector sum.", "Velocity is a vector quantity."],
            "s2": ["Mosquitoes are disease vectors.", "The vector spreads the infection."]
        }
    },
    "colony": {
        "senses": {
            "s1": "A settlement in a new territory.",
            "s2": "A group of microorganisms growing together."
        },
        "domains": ["History", "Biology"],
        "examples": {
            "s1": ["The American colonies declared independence.", "A colony of settlers."],
            "s2": ["Count the bacterial colonies.", "The colony grew overnight."]
        }
    },
    "virus": {
        "senses": {
            "s1": "An infectious agent that replicates inside cells.",
            "s2": "Malicious computer code that replicates."
        },
        "domains": ["Biology", "Technology"],
        "examples": {
            "s1": ["The flu virus spreads easily.", "Viruses cannot reproduce on their own."],
            "s2": ["A virus infected the computer.", "Install antivirus software."]
        }
    },
    "pupil": {
        "senses": {
            "s1": "A student, especially a young one.",
            "s2": "The dark opening in the center of the eye."
        },
        "domains": ["Education", "Anatomy"],
        "examples": {
            "s1": ["The pupil answered correctly.", "A star pupil."],
            "s2": ["The pupil dilates in darkness.", "Check the pupil response."]
        }
    },
    "bridge": {
        "senses": {
            "s1": "A structure crossing an obstacle.",
            "s2": "A dental appliance replacing missing teeth."
        },
        "domains": ["Architecture", "Dental"],
        "examples": {
            "s1": ["The bridge spans the river.", "Cross the bridge carefully."],
            "s2": ["She needs a dental bridge.", "The bridge replaces three teeth."]
        }
    }
}

# =============================================================================
# DOMAIN 3: TECHNOLOGY (50 words)
# =============================================================================

TECHNOLOGY_CATALOG = {
    "bug": {
        "senses": {
            "s1": "A small insect.",
            "s2": "A software error or defect."
        },
        "domains": ["Biology", "Technology"],
        "examples": {
            "s1": ["A bug crawled across the floor.", "Bugs are attracted to light."],
            "s2": ["There's a bug in the code.", "We fixed the critical bug."]
        }
    },
    "cookie": {
        "senses": {
            "s1": "A sweet baked treat.",
            "s2": "A small file stored by a website on a user's computer."
        },
        "domains": ["Food", "Technology"],
        "examples": {
            "s1": ["Would you like a chocolate chip cookie?", "Fresh baked cookies."],
            "s2": ["This site uses cookies.", "Clear your browser cookies."]
        }
    },
    "spam": {
        "senses": {
            "s1": "A canned meat product.",
            "s2": "Unsolicited electronic messages."
        },
        "domains": ["Food", "Technology"],
        "examples": {
            "s1": ["Spam was popular during the war.", "A can of Spam."],
            "s2": ["My inbox is full of spam.", "The spam filter blocked it."]
        }
    },
    "thread": {
        "senses": {
            "s1": "A thin strand of cotton or other fiber.",
            "s2": "A sequence of messages in a forum or email."
        },
        "domains": ["Textiles", "Technology"],
        "examples": {
            "s1": ["The thread broke while sewing.", "A spool of thread."],
            "s2": ["Read the entire thread.", "Start a new thread."]
        }
    },
    "web": {
        "senses": {
            "s1": "A network of fine threads spun by a spider.",
            "s2": "The World Wide Web (internet)."
        },
        "domains": ["Nature", "Technology"],
        "examples": {
            "s1": ["The spider built a web.", "A web of intrigue."],
            "s2": ["Browse the web for information.", "Web development is a career."]
        }
    },
    "pirate": {
        "senses": {
            "s1": "A person who attacks ships at sea.",
            "s2": "One who illegally copies or distributes software/media."
        },
        "domains": ["Maritime", "Technology"],
        "examples": {
            "s1": ["Pirates roamed the Caribbean.", "A pirate ship."],
            "s2": ["Software pirates cost the industry billions.", "Don't pirate movies."]
        }
    },
    "port": {
        "senses": {
            "s1": "A harbor where ships dock.",
            "s2": "A connection point for data transfer."
        },
        "domains": ["Maritime", "Technology"],
        "examples": {
            "s1": ["The ship arrived at port.", "A busy trading port."],
            "s2": ["Connect to port 8080.", "USB port."]
        }
    },
    "bus": {
        "senses": {
            "s1": "A large vehicle for public transport.",
            "s2": "A system for transferring data between components."
        },
        "domains": ["Transportation", "Technology"],
        "examples": {
            "s1": ["Take the bus to work.", "The school bus arrived."],
            "s2": ["The data bus carries information.", "A 64-bit bus."]
        }
    },
    "driver": {
        "senses": {
            "s1": "A person who operates a vehicle.",
            "s2": "Software that controls hardware."
        },
        "domains": ["Transportation", "Technology"],
        "examples": {
            "s1": ["The driver stopped at the light.", "A taxi driver."],
            "s2": ["Update your graphics driver.", "The driver installation failed."]
        }
    },
    "monitor": {
        "senses": {
            "s1": "To observe or watch over.",
            "s2": "A display screen for a computer."
        },
        "domains": ["Surveillance", "Technology"],
        "examples": {
            "s1": ["Monitor the patient's vitals.", "We monitor all activity."],
            "s2": ["Buy a new monitor.", "The monitor resolution is 4K."]
        }
    },
    "chip": {
        "senses": {
            "s1": "A thin slice of potato (British) or a small piece.",
            "s2": "An integrated circuit."
        },
        "domains": ["Food", "Technology"],
        "examples": {
            "s1": ["Fish and chips for dinner.", "A chocolate chip."],
            "s2": ["The chip processes millions of operations.", "A silicon chip."]
        }
    },
    "memory": {
        "senses": {
            "s1": "The faculty of recalling past experiences.",
            "s2": "Computer storage for data."
        },
        "domains": ["Cognitive", "Technology"],
        "examples": {
            "s1": ["Cherish the memory.", "A good memory for names."],
            "s2": ["Add more memory to your computer.", "16GB of RAM memory."]
        }
    },
    "crash": {
        "senses": {
            "s1": "A violent collision.",
            "s2": "A sudden failure of a computer system.",
            "s3": "A sudden drop in market value."
        },
        "domains": ["Accident", "Technology", "Finance"],
        "examples": {
            "s1": ["The car crash was terrible.", "A plane crash."],
            "s2": ["The system crashed again.", "Save often in case of a crash."],
            "s3": ["The stock market crash of 1929.", "Crypto crash wiped out savings."]
        }
    },
    "freeze": {
        "senses": {
            "s1": "To turn to ice or become very cold.",
            "s2": "To become unresponsive (computer)."
        },
        "domains": ["Temperature", "Technology"],
        "examples": {
            "s1": ["Water freezes at 0 degrees.", "Freeze the leftovers."],
            "s2": ["The computer froze during the update.", "The app keeps freezing."]
        }
    },
    "boot": {
        "senses": {
            "s1": "A type of footwear.",
            "s2": "To start up a computer."
        },
        "domains": ["Fashion", "Technology"],
        "examples": {
            "s1": ["Wear boots in the snow.", "Hiking boots."],
            "s2": ["Boot up the computer.", "The system failed to boot."]
        }
    },
    "icon": {
        "senses": {
            "s1": "A religious image venerated in Eastern Christianity.",
            "s2": "A small clickable symbol on a screen."
        },
        "domains": ["Religion", "Technology"],
        "examples": {
            "s1": ["The Byzantine icon depicted saints.", "A holy icon."],
            "s2": ["Click the icon to open the app.", "Desktop icons."]
        }
    },
    "menu": {
        "senses": {
            "s1": "A list of food options at a restaurant.",
            "s2": "A list of commands or options in software."
        },
        "domains": ["Food", "Technology"],
        "examples": {
            "s1": ["Check the lunch menu.", "The menu includes vegan options."],
            "s2": ["Open the File menu.", "Right-click for the context menu."]
        }
    },
    "window": {
        "senses": {
            "s1": "An opening in a wall with glass.",
            "s2": "A rectangular area on a computer screen."
        },
        "domains": ["Architecture", "Technology"],
        "examples": {
            "s1": ["Open the window for fresh air.", "A stained glass window."],
            "s2": ["Open a new browser window.", "Minimize the window."]
        }
    },
    "folder": {
        "senses": {
            "s1": "A cardboard holder for papers.",
            "s2": "A directory in a computer file system."
        },
        "domains": ["Office", "Technology"],
        "examples": {
            "s1": ["Put the documents in a folder.", "A manila folder."],
            "s2": ["Create a new folder on the desktop.", "The folder contains images."]
        }
    },
    "traffic": {
        "senses": {
            "s1": "Vehicles moving on roads.",
            "s2": "Data packets transmitted over a network."
        },
        "domains": ["Transportation", "Technology"],
        "examples": {
            "s1": ["Traffic is heavy at rush hour.", "A traffic jam."],
            "s2": ["Monitor network traffic.", "High traffic crashed the server."]
        }
    },
    "protocol": {
        "senses": {
            "s1": "Official procedure or etiquette.",
            "s2": "Rules for data exchange between systems."
        },
        "domains": ["Diplomacy", "Technology"],
        "examples": {
            "s1": ["Follow protocol at the ceremony.", "Diplomatic protocol."],
            "s2": ["HTTP is a protocol.", "The protocol specifies the format."]
        }
    },
    "platform": {
        "senses": {
            "s1": "A raised flat surface or stage.",
            "s2": "An operating system or environment for software."
        },
        "domains": ["Physical", "Technology"],
        "examples": {
            "s1": ["Wait on the train platform.", "A diving platform."],
            "s2": ["The platform supports multiple languages.", "A social media platform."]
        }
    },
    "code": {
        "senses": {
            "s1": "A system of words or symbols for secrecy.",
            "s2": "Instructions written for a computer."
        },
        "domains": ["Cipher", "Technology"],
        "examples": {
            "s1": ["They communicated in code.", "The Morse code."],
            "s2": ["Write clean code.", "Debug the code."]
        }
    },
    "library": {
        "senses": {
            "s1": "A building or room containing books.",
            "s2": "A collection of pre-written code functions."
        },
        "domains": ["Books", "Technology"],
        "examples": {
            "s1": ["Study at the library.", "A public library."],
            "s2": ["Import the math library.", "Python libraries simplify development."]
        }
    },
    "patch": {
        "senses": {
            "s1": "A piece of material covering a hole.",
            "s2": "A software update fixing bugs."
        },
        "domains": ["Repair", "Technology"],
        "examples": {
            "s1": ["Sew a patch on the jeans.", "An eye patch."],
            "s2": ["Apply the security patch.", "The patch fixed the vulnerability."]
        }
    },
    "token": {
        "senses": {
            "s1": "A small piece representing something.",
            "s2": "A digital unit for authentication or cryptocurrency."
        },
        "domains": ["General", "Technology"],
        "examples": {
            "s1": ["A token of appreciation.", "Arcade tokens."],
            "s2": ["Generate an access token.", "Crypto tokens have value."]
        }
    },
    "key": {
        "senses": {
            "s1": "A metal device for opening locks.",
            "s2": "A code or password for encryption."
        },
        "domains": ["General", "Technology"],
        "examples": {
            "s1": ["I lost my house key.", "The key fits the lock."],
            "s2": ["The encryption key is secret.", "Enter the API key."]
        }
    },
    "hash": {
        "senses": {
            "s1": "Chopped food, especially corned beef hash.",
            "s2": "A fixed-size output from a cryptographic function."
        },
        "domains": ["Food", "Technology"],
        "examples": {
            "s1": ["Order hash browns.", "Corned beef hash for breakfast."],
            "s2": ["Compute the hash of the file.", "SHA-256 produces a secure hash."]
        }
    },
    "root": {
        "senses": {
            "s1": "The part of a plant underground.",
            "s2": "The superuser account or base directory."
        },
        "domains": ["Botany", "Technology"],
        "examples": {
            "s1": ["The roots spread deep.", "Root vegetables."],
            "s2": ["Log in as root.", "Navigate to the root directory."]
        }
    },
    "tree": {
        "senses": {
            "s1": "A perennial woody plant.",
            "s2": "A hierarchical data structure."
        },
        "domains": ["Botany", "Technology"],
        "examples": {
            "s1": ["The tree provides shade.", "An oak tree."],
            "s2": ["Traverse the binary tree.", "The DOM is a tree structure."]
        }
    },
    "branch": {
        "senses": {
            "s1": "A part extending from a tree trunk.",
            "s2": "A divergent version of code in version control."
        },
        "domains": ["Botany", "Technology"],
        "examples": {
            "s1": ["The bird sat on a branch.", "A tree branch."],
            "s2": ["Create a new branch for the feature.", "Merge the branch into main."]
        }
    },
    "shell": {
        "senses": {
            "s1": "The hard outer covering of an animal.",
            "s2": "A command-line interface."
        },
        "domains": ["Biology", "Technology"],
        "examples": {
            "s1": ["The shell protects the turtle.", "A seashell."],
            "s2": ["Open a shell terminal.", "Run the command in the shell."]
        }
    },
    "kernel": {
        "senses": {
            "s1": "The inner part of a seed or nut.",
            "s2": "The core of an operating system."
        },
        "domains": ["Food", "Technology"],
        "examples": {
            "s1": ["A kernel of corn.", "The nut kernel is edible."],
            "s2": ["The Linux kernel manages hardware.", "Update the kernel."]
        }
    },
    "worm": {
        "senses": {
            "s1": "A long soft-bodied invertebrate.",
            "s2": "Self-replicating malware."
        },
        "domains": ["Biology", "Technology"],
        "examples": {
            "s1": ["The early bird catches the worm.", "Earthworms improve soil."],
            "s2": ["The worm spread across the network.", "A computer worm."]
        }
    },
    "trojan": {
        "senses": {
            "s1": "Relating to ancient Troy.",
            "s2": "Malware disguised as legitimate software."
        },
        "domains": ["History", "Technology"],
        "examples": {
            "s1": ["The Trojan War lasted ten years.", "Trojan heroes."],
            "s2": ["A Trojan infected the system.", "Don't download Trojans."]
        }
    },
    "firewall": {
        "senses": {
            "s1": "A wall to prevent fire from spreading.",
            "s2": "Network security that monitors traffic."
        },
        "domains": ["Building", "Technology"],
        "examples": {
            "s1": ["The firewall contained the blaze.", "Building code requires firewalls."],
            "s2": ["The firewall blocked the attack.", "Configure the firewall rules."]
        }
    },
    "gateway": {
        "senses": {
            "s1": "An entrance or archway.",
            "s2": "A network node connecting different networks."
        },
        "domains": ["Architecture", "Technology"],
        "examples": {
            "s1": ["The gateway to the garden.", "An impressive stone gateway."],
            "s2": ["The gateway routes packets.", "Set the default gateway."]
        }
    }
}

# =============================================================================
# DOMAIN 4: SLANG (50 words)
# =============================================================================

SLANG_CATALOG = {
    "tea": {
        "senses": {
            "s1": "A hot beverage made from leaves.",
            "s2": "Gossip or truth (slang)."
        },
        "domains": ["Beverage", "Slang"],
        "examples": {
            "s1": ["Would you like some tea?", "Earl Grey tea is my favorite."],
            "s2": ["Spill the tea!", "That's the tea, sis."]
        }
    },
    "salty": {
        "senses": {
            "s1": "Containing or tasting of salt.",
            "s2": "Bitter, upset, or resentful (slang)."
        },
        "domains": ["Taste", "Slang"],
        "examples": {
            "s1": ["The soup is too salty.", "Ocean water is salty."],
            "s2": ["Why are you so salty?", "Don't be salty about losing."]
        }
    },
    "ghost": {
        "senses": {
            "s1": "The spirit of a dead person.",
            "s2": "To suddenly stop all communication (slang)."
        },
        "domains": ["Supernatural", "Slang"],
        "examples": {
            "s1": ["The ghost haunted the house.", "A scary ghost story."],
            "s2": ["He ghosted me after three dates.", "Don't ghost your friends."]
        }
    },
    "drip": {
        "senses": {
            "s1": "A small drop of liquid falling.",
            "s2": "Stylish clothes or swagger (slang)."
        },
        "domains": ["Liquid", "Slang"],
        "examples": {
            "s1": ["The faucet has a drip.", "A drip of water."],
            "s2": ["Check out his drip!", "That outfit has serious drip."]
        }
    },
    "sick": {
        "senses": {
            "s1": "Affected by illness.",
            "s2": "Extremely cool or impressive (slang)."
        },
        "domains": ["Health", "Slang"],
        "examples": {
            "s1": ["I feel sick today.", "She called in sick."],
            "s2": ["That trick was sick!", "Sick beats, bro."]
        }
    },
    "lit": {
        "senses": {
            "s1": "Illuminated by light.",
            "s2": "Exciting, amazing, or intoxicated (slang)."
        },
        "domains": ["Light", "Slang"],
        "examples": {
            "s1": ["The candles were lit.", "A well-lit room."],
            "s2": ["The party was lit!", "We got lit last night."]
        }
    },
    "basic": {
        "senses": {
            "s1": "Fundamental or essential.",
            "s2": "Unoriginal or mainstream in a boring way (slang)."
        },
        "domains": ["General", "Slang"],
        "examples": {
            "s1": ["Learn the basic skills first.", "Basic training."],
            "s2": ["Pumpkin spice lattes are so basic.", "She's basic."]
        }
    },
    "extra": {
        "senses": {
            "s1": "Additional; more than usual.",
            "s2": "Overly dramatic or excessive (slang)."
        },
        "domains": ["General", "Slang"],
        "examples": {
            "s1": ["I need an extra pillow.", "Extra cheese, please."],
            "s2": ["Stop being so extra.", "That reaction was extra."]
        }
    },
    "flex": {
        "senses": {
            "s1": "To bend a muscle.",
            "s2": "To show off or boast (slang)."
        },
        "domains": ["Physical", "Slang"],
        "examples": {
            "s1": ["Flex your bicep.", "Muscles flex during exercise."],
            "s2": ["Weird flex but okay.", "He's always flexing his money."]
        }
    },
    "shade": {
        "senses": {
            "s1": "An area of darkness from blocked light.",
            "s2": "Subtle insult or disrespect (slang)."
        },
        "domains": ["Light", "Slang"],
        "examples": {
            "s1": ["Sit in the shade.", "The tree provides shade."],
            "s2": ["She threw shade at him.", "No shade intended."]
        }
    },
    "stan": {
        "senses": {
            "s1": "A common first name.",
            "s2": "An extremely devoted fan (slang)."
        },
        "domains": ["Name", "Slang"],
        "examples": {
            "s1": ["His name is Stan.", "Stan is my neighbor."],
            "s2": ["I stan this artist.", "The stans crashed the server."]
        }
    },
    "goat": {
        "senses": {
            "s1": "A horned mammal.",
            "s2": "Greatest Of All Time (slang acronym)."
        },
        "domains": ["Animal", "Slang"],
        "examples": {
            "s1": ["The goat climbed the hill.", "Goat cheese."],
            "s2": ["Michael Jordan is the GOAT.", "She's the GOAT of tennis."]
        }
    },
    "bread": {
        "senses": {
            "s1": "A baked food made from flour.",
            "s2": "Money (slang)."
        },
        "domains": ["Food", "Slang"],
        "examples": {
            "s1": ["Fresh bread from the bakery.", "Slice the bread."],
            "s2": ["I need to make some bread.", "Get that bread!"]
        }
    },
    "ice": {
        "senses": {
            "s1": "Frozen water.",
            "s2": "Diamonds or jewelry (slang)."
        },
        "domains": ["Temperature", "Slang"],
        "examples": {
            "s1": ["Put ice in the drink.", "Ice melts at 0°C."],
            "s2": ["Check out his ice!", "Wearing ice on her wrist."]
        }
    },
    "heat": {
        "senses": {
            "s1": "The quality of being hot.",
            "s2": "A gun; police attention (slang)."
        },
        "domains": ["Temperature", "Slang"],
        "examples": {
            "s1": ["The heat is unbearable.", "Turn up the heat."],
            "s2": ["He's packing heat.", "The heat is coming down."]
        }
    },
    "beef": {
        "senses": {
            "s1": "Meat from cattle.",
            "s2": "A conflict or grudge (slang)."
        },
        "domains": ["Food", "Slang"],
        "examples": {
            "s1": ["Beef is high in protein.", "Ground beef."],
            "s2": ["They have beef.", "Squash the beef."]
        }
    },
    "rat": {
        "senses": {
            "s1": "A rodent larger than a mouse.",
            "s2": "An informant or snitch (slang)."
        },
        "domains": ["Animal", "Slang"],
        "examples": {
            "s1": ["The rat scurried away.", "Rats spread disease."],
            "s2": ["He ratted us out.", "Don't be a rat."]
        }
    },
    "snake": {
        "senses": {
            "s1": "A legless reptile.",
            "s2": "A deceitful or treacherous person (slang)."
        },
        "domains": ["Animal", "Slang"],
        "examples": {
            "s1": ["The snake slithered away.", "A venomous snake."],
            "s2": ["She's such a snake.", "Watch out for snakes in the office."]
        }
    },
    "bomb": {
        "senses": {
            "s1": "An explosive device.",
            "s2": "To fail spectacularly (US slang); to succeed greatly (UK slang)."
        },
        "domains": ["Military", "Slang"],
        "examples": {
            "s1": ["The bomb was defused.", "A nuclear bomb."],
            "s2": ["The movie bombed at the box office.", "That performance was the bomb! (UK)"]
        }
    },
    "smash": {
        "senses": {
            "s1": "To break violently into pieces.",
            "s2": "A sexual encounter (slang)."
        },
        "domains": ["Action", "Slang"],
        "examples": {
            "s1": ["Smash the piñata.", "The glass smashed on the floor."],
            "s2": ["Smash or pass?", "They smashed."]
        }
    },
    "slap": {
        "senses": {
            "s1": "A sharp blow with the hand.",
            "s2": "To be excellent, especially music (slang)."
        },
        "domains": ["Action", "Slang"],
        "examples": {
            "s1": ["She slapped his face.", "A slap on the wrist."],
            "s2": ["This song slaps!", "That beat slaps hard."]
        }
    },
    "kicks": {
        "senses": {
            "s1": "Striking movements with the foot.",
            "s2": "Sneakers or shoes (slang)."
        },
        "domains": ["Action", "Slang"],
        "examples": {
            "s1": ["Martial arts kicks.", "The horse kicks."],
            "s2": ["Nice kicks, bro.", "Fresh kicks from the store."]
        }
    },
    "sauce": {
        "senses": {
            "s1": "A liquid condiment for food.",
            "s2": "Style, confidence, or swagger (slang)."
        },
        "domains": ["Food", "Slang"],
        "examples": {
            "s1": ["Add more sauce.", "Tomato sauce."],
            "s2": ["He's got the sauce.", "That outfit has sauce."]
        }
    },
    "juice": {
        "senses": {
            "s1": "Liquid extracted from fruit.",
            "s2": "Power, influence, or steroids (slang)."
        },
        "domains": ["Beverage", "Slang"],
        "examples": {
            "s1": ["Orange juice for breakfast.", "Fresh juice."],
            "s2": ["He's got juice in this town.", "On the juice (steroids)."]
        }
    },
    "slay": {
        "senses": {
            "s1": "To kill violently.",
            "s2": "To do something extremely well (slang)."
        },
        "domains": ["Violence", "Slang"],
        "examples": {
            "s1": ["The knight slew the dragon.", "Slain in battle."],
            "s2": ["You slayed that performance!", "Slay, queen!"]
        }
    },
    "dead": {
        "senses": {
            "s1": "No longer alive.",
            "s2": "Finding something extremely funny (slang)."
        },
        "domains": ["Life", "Slang"],
        "examples": {
            "s1": ["The plant is dead.", "Dead silence."],
            "s2": ["I'm dead! 💀", "That joke had me dead."]
        }
    },
    "sleep": {
        "senses": {
            "s1": "A state of rest with reduced consciousness.",
            "s2": "To ignore or underestimate (slang)."
        },
        "domains": ["Rest", "Slang"],
        "examples": {
            "s1": ["I need more sleep.", "Sleep well."],
            "s2": ["Don't sleep on this artist.", "They're sleeping on her talent."]
        }
    },
    "sus": {
        "senses": {
            "s1": "Abbreviation for suspicious.",
            "s2": "Behaving in a questionable way (slang from Among Us)."
        },
        "domains": ["Slang", "Gaming"],
        "examples": {
            "s1": ["That story sounds sus.", "Something's sus."],
            "s2": ["Red is sus.", "Acting pretty sus right now."]
        }
    }
}

# =============================================================================
# DOMAIN 5: GENERAL & ACADEMIC (50 words)
# =============================================================================

GENERAL_ACADEMIC_CATALOG = {
    "head": {
        "senses": {
            "s1": "The upper part of the body.",
            "s2": "A leader or person in charge.",
            "s3": "The source of a river."
        },
        "domains": ["Anatomy", "Leadership", "Geography"],
        "examples": {
            "s1": ["My head hurts.", "Nod your head."],
            "s2": ["The head of the department.", "Head of state."],
            "s3": ["The head of the river.", "At the head of the valley."]
        }
    },
    "face": {
        "senses": {
            "s1": "The front of the head.",
            "s2": "To confront or deal with.",
            "s3": "A surface of an object."
        },
        "domains": ["Anatomy", "Action", "Geometry"],
        "examples": {
            "s1": ["A smile on her face.", "Face to face."],
            "s2": ["Face the truth.", "Face your fears."],
            "s3": ["The face of the clock.", "A cube has six faces."]
        }
    },
    "foot": {
        "senses": {
            "s1": "The lower extremity of the leg.",
            "s2": "A unit of measurement (12 inches).",
            "s3": "The base or bottom of something."
        },
        "domains": ["Anatomy", "Measurement", "Position"],
        "examples": {
            "s1": ["My foot is sore.", "Left foot forward."],
            "s2": ["Six feet tall.", "Measure in feet."],
            "s3": ["At the foot of the mountain.", "The foot of the bed."]
        }
    },
    "arm": {
        "senses": {
            "s1": "The upper limb of the body.",
            "s2": "A weapon or to equip with weapons.",
            "s3": "A branch of an organization."
        },
        "domains": ["Anatomy", "Military", "Organization"],
        "examples": {
            "s1": ["Raise your arm.", "A broken arm."],
            "s2": ["The right to bear arms.", "Arm the troops."],
            "s3": ["The research arm of the company.", "An arm of the government."]
        }
    },
    "eye": {
        "senses": {
            "s1": "The organ of sight.",
            "s2": "The hole in a needle.",
            "s3": "The center of a storm."
        },
        "domains": ["Anatomy", "Sewing", "Weather"],
        "examples": {
            "s1": ["Blue eyes.", "Keep an eye on it."],
            "s2": ["Thread through the eye.", "The eye of the needle."],
            "s3": ["In the eye of the hurricane.", "The storm's eye passed over."]
        }
    },
    "mouth": {
        "senses": {
            "s1": "The opening for food and speech.",
            "s2": "The place where a river enters the sea.",
            "s3": "The entrance to a cave."
        },
        "domains": ["Anatomy", "Geography"],
        "examples": {
            "s1": ["Open your mouth.", "Don't talk with your mouth full."],
            "s2": ["The mouth of the Mississippi.", "At the river's mouth."],
            "s3": ["The mouth of the cave.", "Enter through the mouth."]
        }
    },
    "degree": {
        "senses": {
            "s1": "A unit of temperature or angle measurement.",
            "s2": "An academic qualification.",
            "s3": "The extent or intensity of something."
        },
        "domains": ["Measurement", "Education", "General"],
        "examples": {
            "s1": ["90 degrees Fahrenheit.", "A 45-degree angle."],
            "s2": ["She earned a degree in physics.", "Bachelor's degree."],
            "s3": ["A high degree of difficulty.", "To what degree?"]
        }
    },
    "grade": {
        "senses": {
            "s1": "A level of quality or rank.",
            "s2": "A mark indicating achievement.",
            "s3": "A slope or incline."
        },
        "domains": ["Quality", "Education", "Geography"],
        "examples": {
            "s1": ["Top grade materials.", "What grade are you in?"],
            "s2": ["She got an A grade.", "Grades will be posted Friday."],
            "s3": ["A steep grade on the road.", "The grade was too steep for trucks."]
        }
    },
    "mark": {
        "senses": {
            "s1": "A visible impression or stain.",
            "s2": "A grade or score.",
            "s3": "A target or goal."
        },
        "domains": ["Visual", "Education", "Target"],
        "examples": {
            "s1": ["The mark won't come off.", "Leave a mark."],
            "s2": ["Full marks on the test.", "She got high marks."],
            "s3": ["Hit the mark.", "On your mark, get set, go!"]
        }
    },
    "chair": {
        "senses": {
            "s1": "A seat with a back.",
            "s2": "The person leading a meeting or committee."
        },
        "domains": ["Furniture", "Leadership"],
        "examples": {
            "s1": ["Sit in the chair.", "A wooden chair."],
            "s2": ["The chair called the meeting to order.", "She's the department chair."]
        }
    },
    "course": {
        "senses": {
            "s1": "A path or direction.",
            "s2": "A series of educational lessons.",
            "s3": "A part of a meal."
        },
        "domains": ["Navigation", "Education", "Food"],
        "examples": {
            "s1": ["The ship changed course.", "Stay the course."],
            "s2": ["Take a chemistry course.", "Online courses."],
            "s3": ["The main course.", "A five-course meal."]
        }
    },
    "class": {
        "senses": {
            "s1": "A group of students taught together.",
            "s2": "A social or economic category.",
            "s3": "A level of quality."
        },
        "domains": ["Education", "Society", "Quality"],
        "examples": {
            "s1": ["The class starts at 9 AM.", "Class dismissed."],
            "s2": ["Middle class.", "Class struggle."],
            "s3": ["First class service.", "World class athlete."]
        }
    },
    "major": {
        "senses": {
            "s1": "Of great importance or significance.",
            "s2": "A field of study in college.",
            "s3": "A military rank."
        },
        "domains": ["General", "Education", "Military"],
        "examples": {
            "s1": ["A major breakthrough.", "Major changes."],
            "s2": ["She's a biology major.", "What's your major?"],
            "s3": ["Major Smith reported for duty.", "Promoted to major."]
        }
    },
    "minor": {
        "senses": {
            "s1": "Lesser in importance or size.",
            "s2": "A secondary field of study.",
            "s3": "A person under legal age."
        },
        "domains": ["General", "Education", "Legal"],
        "examples": {
            "s1": ["A minor issue.", "Minor adjustments."],
            "s2": ["She has a minor in French.", "Declare a minor."],
            "s3": ["Minors cannot purchase alcohol.", "A legal minor."]
        }
    },
    "line": {
        "senses": {
            "s1": "A long narrow mark or band.",
            "s2": "A row of people waiting.",
            "s3": "A telephone connection."
        },
        "domains": ["Geometry", "Queue", "Communication"],
        "examples": {
            "s1": ["Draw a straight line.", "The line extends to the horizon."],
            "s2": ["Wait in line.", "The line was long."],
            "s3": ["The line is busy.", "Hold the line."]
        }
    },
    "column": {
        "senses": {
            "s1": "A vertical pillar.",
            "s2": "A vertical arrangement of text.",
            "s3": "A regular newspaper article."
        },
        "domains": ["Architecture", "Layout", "Media"],
        "examples": {
            "s1": ["Marble columns.", "A column supporting the roof."],
            "s2": ["Add a column to the spreadsheet.", "Data in the column."],
            "s3": ["She writes a weekly column.", "The sports column."]
        }
    },
    "row": {
        "senses": {
            "s1": "A horizontal line of items.",
            "s2": "To propel a boat with oars.",
            "s3": "A noisy argument (British)."
        },
        "domains": ["Layout", "Action", "Conflict"],
        "examples": {
            "s1": ["Front row seats.", "Row three of the spreadsheet."],
            "s2": ["Row the boat.", "They rowed across the lake."],
            "s3": ["They had a terrible row.", "A family row."]
        }
    },
    "sheet": {
        "senses": {
            "s1": "Bed linen.",
            "s2": "A flat piece of paper.",
            "s3": "A large expanse of something."
        },
        "domains": ["Household", "Office", "Geography"],
        "examples": {
            "s1": ["Change the sheets.", "Clean sheets."],
            "s2": ["A sheet of paper.", "Sign the sheet."],
            "s3": ["A sheet of ice.", "Sheets of rain."]
        }
    },
    "board": {
        "senses": {
            "s1": "A flat piece of wood.",
            "s2": "A group of directors.",
            "s3": "Meals provided regularly."
        },
        "domains": ["Material", "Corporate", "Hospitality"],
        "examples": {
            "s1": ["A wooden board.", "Cut on the board."],
            "s2": ["The board approved the merger.", "Board of directors."],
            "s3": ["Room and board.", "Full board included."]
        }
    },
    "cabinet": {
        "senses": {
            "s1": "A cupboard for storage.",
            "s2": "A group of senior government ministers."
        },
        "domains": ["Furniture", "Government"],
        "examples": {
            "s1": ["The kitchen cabinet.", "Filing cabinet."],
            "s2": ["The cabinet met today.", "Cabinet reshuffle."]
        }
    },
    "floor": {
        "senses": {
            "s1": "The surface of a room.",
            "s2": "A story of a building.",
            "s3": "The right to speak in a debate."
        },
        "domains": ["Architecture", "Building", "Parliamentary"],
        "examples": {
            "s1": ["Clean the floor.", "Hardwood floor."],
            "s2": ["Third floor.", "Floor plan."],
            "s3": ["The senator has the floor.", "Yield the floor."]
        }
    },
    "house": {
        "senses": {
            "s1": "A building for dwelling.",
            "s2": "A legislative assembly.",
            "s3": "A business establishment."
        },
        "domains": ["Architecture", "Government", "Business"],
        "examples": {
            "s1": ["Buy a house.", "House for sale."],
            "s2": ["The House voted today.", "House of Representatives."],
            "s3": ["A publishing house.", "On the house (free)."]
        }
    },
    "bill": {
        "senses": {
            "s1": "A bird's beak.",
            "s2": "A proposed law.",
            "s3": "A statement of charges."
        },
        "domains": ["Biology", "Government", "Commerce"],
        "examples": {
            "s1": ["The duck's bill.", "A yellow bill."],
            "s2": ["The bill passed the Senate.", "Draft a bill."],
            "s3": ["Pay the bill.", "The bill came to $50."]
        }
    },
    "act": {
        "senses": {
            "s1": "A deed or action.",
            "s2": "A law passed by legislature.",
            "s3": "A segment of a play."
        },
        "domains": ["Action", "Legal", "Theater"],
        "examples": {
            "s1": ["An act of kindness.", "Caught in the act."],
            "s2": ["The Civil Rights Act.", "Pass an act."],
            "s3": ["Act one, scene two.", "The final act."]
        }
    },
    "passage": {
        "senses": {
            "s1": "A corridor or hallway.",
            "s2": "A section of text.",
            "s3": "The act of traveling through."
        },
        "domains": ["Architecture", "Literature", "Travel"],
        "examples": {
            "s1": ["A dark passage.", "The secret passage."],
            "s2": ["Read the passage aloud.", "A biblical passage."],
            "s3": ["Safe passage.", "The passage of time."]
        }
    },
    "chapter": {
        "senses": {
            "s1": "A main division of a book.",
            "s2": "A local branch of an organization.",
            "s3": "A period in life."
        },
        "domains": ["Literature", "Organization", "Life"],
        "examples": {
            "s1": ["Read chapter five.", "The final chapter."],
            "s2": ["The local chapter.", "Start a chapter."],
            "s3": ["A new chapter in life.", "Close that chapter."]
        }
    },
    "legend": {
        "senses": {
            "s1": "A traditional story or myth.",
            "s2": "A key explaining symbols on a map."
        },
        "domains": ["Folklore", "Cartography"],
        "examples": {
            "s1": ["The legend of King Arthur.", "A living legend."],
            "s2": ["Check the legend.", "The map legend explains colors."]
        }
    },
    "note": {
        "senses": {
            "s1": "A brief written message.",
            "s2": "A musical sound.",
            "s3": "A piece of paper money."
        },
        "domains": ["Writing", "Music", "Finance"],
        "examples": {
            "s1": ["Leave a note.", "Take notes in class."],
            "s2": ["Hit the high note.", "A musical note."],
            "s3": ["A ten-dollar note.", "Bank notes."]
        }
    },
    "rest": {
        "senses": {
            "s1": "Relaxation or sleep.",
            "s2": "The remaining part.",
            "s3": "A musical silence."
        },
        "domains": ["Health", "General", "Music"],
        "examples": {
            "s1": ["Get some rest.", "A day of rest."],
            "s2": ["The rest of the story.", "Keep the rest."],
            "s3": ["A quarter rest.", "Count the rests."]
        }
    },
    "flat": {
        "senses": {
            "s1": "Level and even.",
            "s2": "An apartment (British).",
            "s3": "A musical note lowered by a semitone."
        },
        "domains": ["Surface", "Housing", "Music"],
        "examples": {
            "s1": ["A flat surface.", "The tire is flat."],
            "s2": ["Rent a flat.", "A two-bedroom flat."],
            "s3": ["B flat.", "Play it flat."]
        }
    },
    "sharp": {
        "senses": {
            "s1": "Having a cutting edge.",
            "s2": "Intelligent or quick-witted.",
            "s3": "A musical note raised by a semitone."
        },
        "domains": ["Physical", "Mental", "Music"],
        "examples": {
            "s1": ["A sharp knife.", "Sharp edges."],
            "s2": ["A sharp mind.", "Sharp as a tack."],
            "s3": ["F sharp.", "You're playing sharp."]
        }
    },
    "natural": {
        "senses": {
            "s1": "Existing in nature; not artificial.",
            "s2": "A person with innate talent.",
            "s3": "A musical note neither sharp nor flat."
        },
        "domains": ["Nature", "Talent", "Music"],
        "examples": {
            "s1": ["Natural ingredients.", "A natural disaster."],
            "s2": ["She's a natural.", "A natural athlete."],
            "s3": ["Play the natural.", "Return to natural."]
        }
    },
    "measure": {
        "senses": {
            "s1": "To determine size or quantity.",
            "s2": "A bar of music.",
            "s3": "A legislative action."
        },
        "domains": ["Measurement", "Music", "Government"],
        "examples": {
            "s1": ["Measure the length.", "Take measurements."],
            "s2": ["Four beats per measure.", "The first measure."],
            "s3": ["Pass the measure.", "A cost-cutting measure."]
        }
    }
}

# =============================================================================
# MERGE ALL CATALOGS
# =============================================================================

FULL_CATALOG = {}
FULL_CATALOG.update(LAW_FINANCE_CATALOG)
FULL_CATALOG.update(SCIENCE_MEDICAL_CATALOG)
FULL_CATALOG.update(TECHNOLOGY_CATALOG)
FULL_CATALOG.update(SLANG_CATALOG)
FULL_CATALOG.update(GENERAL_ACADEMIC_CATALOG)

# Add words from original catalog that aren't duplicated
ORIGINAL_CATALOG = {
    "bank": {
        "senses": {
            "s1": "A financial institution that accepts deposits.",
            "s2": "The sloping land beside a body of water.",
            "s3": "A row or tier of similar objects."
        },
        "domains": ["Finance", "Geography"],
        "examples": {
            "s1": ["The bank approved my loan.", "Deposit at the bank."],
            "s2": ["We sat on the river bank.", "The bank was grassy."],
            "s3": ["A bank of switches.", "A bank of clouds."]
        }
    },
    "bat": {
        "senses": {
            "s1": "A nocturnal flying mammal.",
            "s2": "A wooden club used in sports."
        },
        "domains": ["Biology", "Sports"],
        "examples": {
            "s1": ["The bat flew at dusk.", "Bats use echolocation."],
            "s2": ["Swing the bat.", "A cricket bat."]
        }
    },
    "date": {
        "senses": {
            "s1": "A specific day on the calendar.",
            "s2": "A romantic social engagement.",
            "s3": "The edible fruit of a palm tree."
        },
        "domains": ["Time", "Social", "Food"],
        "examples": {
            "s1": ["What's the date?", "The expiration date."],
            "s2": ["A dinner date.", "First date."],
            "s3": ["Dried dates.", "Date palm."]
        }
    },
    "match": {
        "senses": {
            "s1": "A competitive game.",
            "s2": "A stick for igniting fire.",
            "s3": "Something equal or corresponding."
        },
        "domains": ["Sports", "General"],
        "examples": {
            "s1": ["The tennis match.", "Win the match."],
            "s2": ["Strike a match.", "Light a match."],
            "s3": ["A perfect match.", "Match the colors."]
        }
    },
    "ring": {
        "senses": {
            "s1": "Circular jewelry worn on the finger.",
            "s2": "The sound of a bell.",
            "s3": "A criminal group working together."
        },
        "domains": ["Fashion", "Sound", "Crime"],
        "examples": {
            "s1": ["A diamond ring.", "Wedding ring."],
            "s2": ["The phone's ring.", "Give me a ring."],
            "s3": ["A drug ring.", "Crime ring."]
        }
    },
    "spring": {
        "senses": {
            "s1": "The season between winter and summer.",
            "s2": "A coiled metal device.",
            "s3": "A natural source of water."
        },
        "domains": ["Time", "Mechanics", "Geography"],
        "examples": {
            "s1": ["Flowers bloom in spring.", "Spring cleaning."],
            "s2": ["The spring broke.", "A coil spring."],
            "s3": ["A hot spring.", "Natural spring water."]
        }
    },
    "toast": {
        "senses": {
            "s1": "Bread browned by heat.",
            "s2": "A tribute with raised glasses.",
            "s3": "To be in serious trouble (slang)."
        },
        "domains": ["Food", "Social", "Slang"],
        "examples": {
            "s1": ["Butter the toast.", "Toast for breakfast."],
            "s2": ["Raise a toast.", "A toast to the bride."],
            "s3": ["We're toast!", "You're toast if caught."]
        }
    }
}

FULL_CATALOG.update(ORIGINAL_CATALOG)


def generate_bundle(word: str, target_sense: str, difficulty: float = 0.5) -> dict:
    """Generate a single polysemy bundle for a word and target sense."""
    if word not in FULL_CATALOG:
        return None

    catalog = FULL_CATALOG[word]
    senses = catalog["senses"]
    examples = catalog["examples"]
    domains = catalog.get("domains", ["General"])

    if target_sense not in senses:
        return None

    target_examples = examples.get(target_sense, [])
    if len(target_examples) < 2:
        return None

    other_senses = [s for s in senses.keys() if s != target_sense]
    if not other_senses:
        return None

    negative_sense = random.choice(other_senses)
    negative_examples = examples.get(negative_sense, [])
    if not negative_examples:
        return None

    bundle_id = f"SPv2_{word.upper()}_{target_sense.upper()}_{random.randint(1000, 9999)}"

    anchor = random.choice(target_examples)
    positive = [ex for ex in target_examples if ex != anchor]
    positive_text = positive[0] if positive else target_examples[0]
    negative_text = random.choice(negative_examples)
    hard_negative = negative_text

    bundle = {
        "bundle_id": bundle_id,
        "word": word,
        "sense_catalog": senses,
        "metadata": {
            "domains": domains,
            "difficulty_score": difficulty,
            "polysemy_count": len(senses),
            "target_sense": target_sense,
            "target_definition": senses[target_sense]
        },
        "pairings": {
            "positive_pair": ["anchor", "positive"],
            "negative_pair": ["anchor", "negative"],
            "hard_negative_pair": ["anchor", "hard_negative"]
        },
        "items": {
            "anchor": anchor,
            "positive": positive_text,
            "negative": negative_text,
            "hard_negative": hard_negative
        }
    }

    return bundle


def generate_all_bundles(bundles_per_word: int = 4) -> List[dict]:
    """Generate bundles for all words in the catalog."""
    all_bundles = []

    for word, catalog in FULL_CATALOG.items():
        senses = list(catalog["senses"].keys())

        for _ in range(bundles_per_word):
            target_sense = random.choice(senses)
            difficulty = random.uniform(0.3, 0.9)

            bundle = generate_bundle(word, target_sense, difficulty)
            if bundle:
                all_bundles.append(bundle)

    return all_bundles


def save_bundles(bundles: List[dict], output_path: Path):
    """Save bundles to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for bundle in bundles:
            f.write(json.dumps(bundle, ensure_ascii=False) + '\n')

    print(f"Saved {len(bundles)} bundles to {output_path}")


def main():
    """Generate and save polysemy bundles."""
    print("SemanticPhase v2 Expanded Bundle Generator")
    print("=" * 60)
    print(f"Total words in catalog: {len(FULL_CATALOG)}")

    # Show breakdown by domain
    print(f"\nDomain breakdown:")
    print(f"  Law/Finance: {len(LAW_FINANCE_CATALOG)} words")
    print(f"  Science/Medical: {len(SCIENCE_MEDICAL_CATALOG)} words")
    print(f"  Technology: {len(TECHNOLOGY_CATALOG)} words")
    print(f"  Slang: {len(SLANG_CATALOG)} words")
    print(f"  General/Academic: {len(GENERAL_ACADEMIC_CATALOG)} words")
    print(f"  Original additions: {len(ORIGINAL_CATALOG)} words")

    # Generate bundles
    random.seed(42)
    bundles = generate_all_bundles(bundles_per_word=4)

    print(f"\nGenerated {len(bundles)} bundles")

    # Save to output
    output_path = Path("paradigm_factory/output/semanticphase_v2_expanded.jsonl")
    save_bundles(bundles, output_path)

    # Show sample
    print("\nSample bundle:")
    print(json.dumps(bundles[0], indent=2))

    return bundles


if __name__ == "__main__":
    main()
