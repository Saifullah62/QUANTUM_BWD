#!/usr/bin/env python3
"""
SemanticPhase v2 Bundle Generator
=================================

Generates high-quality polysemy bundles following the v2 schema specification.
Uses the 125-word catalog from the SemanticPhase framework.

Schema: bundle_id, word, sense_catalog, metadata, pairings, items
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from datetime import datetime

# =============================================================================
# WORD CATALOG: 125 High-Volatility Polysemes
# =============================================================================

WORD_CATALOG = {
    # Cluster A: Homonyms (Distinct Origins)
    "bank": {
        "senses": {
            "s1": "A financial institution that accepts deposits and channels money.",
            "s2": "The sloping land beside a body of water.",
            "s3": "A row or tier of similar objects."
        },
        "domains": ["Finance", "Geography"],
        "examples": {
            "s1": ["The bank approved my loan application.", "I need to deposit money at the bank."],
            "s2": ["We sat on the river bank watching the sunset.", "The bank was covered with wildflowers."],
            "s3": ["A bank of switches controlled the lighting.", "The bank of clouds moved across the sky."]
        }
    },
    "bat": {
        "senses": {
            "s1": "A nocturnal flying mammal.",
            "s2": "A wooden club used to hit a ball in sports.",
            "s3": "To strike or hit something."
        },
        "domains": ["Biology", "Sports"],
        "examples": {
            "s1": ["The bat flew out of the cave at dusk.", "Bats use echolocation to navigate."],
            "s2": ["He swung the bat and hit a home run.", "The cricket bat was made of willow."],
            "s3": ["She didn't bat an eye at the news.", "He batted the ball across the field."]
        }
    },
    "bond": {
        "senses": {
            "s1": "A fixed-income instrument representing a loan to a borrower.",
            "s2": "A strong connection or feeling of friendship.",
            "s3": "A physical restraint or chemical connection."
        },
        "domains": ["Finance", "Personal"],
        "examples": {
            "s1": ["The company issued bonds to raise capital.", "Treasury bonds are considered safe investments."],
            "s2": ["The bond between mother and child is unbreakable.", "They formed a strong bond during training."],
            "s3": ["The molecular bond was extremely stable.", "The prisoner broke free from his bonds."]
        }
    },
    "bar": {
        "senses": {
            "s1": "The legal profession or collective body of lawyers.",
            "s2": "A counter where alcoholic drinks are served.",
            "s3": "A rigid piece of metal or wood.",
            "s4": "To prohibit or prevent."
        },
        "domains": ["Law", "Hospitality"],
        "examples": {
            "s1": ["She passed the bar exam on her first attempt.", "He was admitted to the bar in 2020."],
            "s2": ["They met at the bar after work.", "The bar serves craft cocktails."],
            "s3": ["The iron bar was too heavy to lift.", "A bar of gold was worth millions."],
            "s4": ["They barred him from entering the club.", "Nothing could bar her from success."]
        }
    },
    "cap": {
        "senses": {
            "s1": "To lie or exaggerate (slang: no cap means truth).",
            "s2": "A type of hat with a visor.",
            "s3": "An upper limit or maximum."
        },
        "domains": ["Slang", "Fashion", "Finance"],
        "examples": {
            "s1": ["No cap, that was the best meal ever.", "He's always capping about his achievements."],
            "s2": ["He wore a baseball cap to the game.", "The graduation cap fell off during the ceremony."],
            "s3": ["The salary cap limits team spending.", "There's a cap on how much you can contribute."]
        }
    },
    "cloud": {
        "senses": {
            "s1": "Visible mass of water droplets in the atmosphere.",
            "s2": "Remote server storage and computing services.",
            "s3": "Something that obscures or causes gloom."
        },
        "domains": ["Weather", "Technology"],
        "examples": {
            "s1": ["Dark clouds gathered before the storm.", "The cloud looked like a rabbit."],
            "s2": ["Store your files in the cloud for backup.", "Cloud computing has transformed business."],
            "s3": ["A cloud of suspicion hung over the investigation.", "Her mood was under a cloud."]
        }
    },
    "current": {
        "senses": {
            "s1": "The present time; happening now.",
            "s2": "The flow of electricity.",
            "s3": "A body of water or air moving in a definite direction."
        },
        "domains": ["Time", "Physics", "Geography"],
        "examples": {
            "s1": ["The current situation requires immediate action.", "What are the current trends in fashion?"],
            "s2": ["The electrical current was too strong.", "Alternating current powers most homes."],
            "s3": ["The river current swept the boat downstream.", "Ocean currents affect global climate."]
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
            "s1": ["What date is the meeting scheduled for?", "The expiration date is next month."],
            "s2": ["They went on their first date last Friday.", "It was a dinner date at an Italian restaurant."],
            "s3": ["Dates are a popular snack in the Middle East.", "The date palm grows in desert climates."]
        }
    },
    "draft": {
        "senses": {
            "s1": "A preliminary version of a document.",
            "s2": "A current of cool air.",
            "s3": "The selection of players for sports teams.",
            "s4": "A written order for payment."
        },
        "domains": ["Writing", "Sports", "Finance"],
        "examples": {
            "s1": ["This is just the first draft of the report.", "Can you review my draft proposal?"],
            "s2": ["Close the window, there's a draft.", "The draft made the candle flicker."],
            "s3": ["He was selected in the first round of the NFL draft.", "The draft picks were announced."],
            "s4": ["She paid by bank draft.", "The draft was drawn on a London bank."]
        }
    },
    "fire": {
        "senses": {
            "s1": "The process of combustion; flames.",
            "s2": "To terminate someone's employment.",
            "s3": "Excellent or amazing (slang)."
        },
        "domains": ["General", "Employment", "Slang"],
        "examples": {
            "s1": ["The fire spread quickly through the building.", "We sat around the campfire."],
            "s2": ["The company had to fire 50 employees.", "He was fired for misconduct."],
            "s3": ["That new song is absolute fire.", "Her outfit is fire today."]
        }
    },
    "interest": {
        "senses": {
            "s1": "Money paid for the use of money lent.",
            "s2": "A feeling of curiosity or attention.",
            "s3": "A stake or share in something."
        },
        "domains": ["Finance", "Psychology"],
        "examples": {
            "s1": ["The loan accrues 5% interest annually.", "Credit cards charge high interest rates."],
            "s2": ["She showed great interest in the project.", "The story piqued my interest."],
            "s3": ["He has a financial interest in the company.", "Conflicts of interest must be disclosed."]
        }
    },
    "lead": {
        "senses": {
            "s1": "To guide or direct others.",
            "s2": "A heavy metallic element (Pb).",
            "s3": "The main role in a performance.",
            "s4": "A piece of information suggesting direction."
        },
        "domains": ["Management", "Chemistry", "Entertainment"],
        "examples": {
            "s1": ["She was chosen to lead the team.", "He leads by example."],
            "s2": ["Lead pipes were replaced due to toxicity.", "The bullets were made of lead."],
            "s3": ["He played the lead in the Broadway show.", "She got the lead role."],
            "s4": ["The detective followed up on the lead.", "A promising lead emerged."]
        }
    },
    "match": {
        "senses": {
            "s1": "A competitive game or contest.",
            "s2": "A small stick used to ignite fire.",
            "s3": "Something equal or corresponding."
        },
        "domains": ["Sports", "General"],
        "examples": {
            "s1": ["The tennis match lasted three hours.", "It was an evenly matched game."],
            "s2": ["Strike the match to light the candle.", "The matchbox was nearly empty."],
            "s3": ["Find a match for this sock.", "They're a perfect match for each other."]
        }
    },
    "mouse": {
        "senses": {
            "s1": "A small rodent.",
            "s2": "A hand-held device for controlling a computer cursor."
        },
        "domains": ["Biology", "Technology"],
        "examples": {
            "s1": ["The mouse scurried across the floor.", "The cat caught a mouse."],
            "s2": ["Click the mouse button twice.", "My wireless mouse needs new batteries."]
        }
    },
    "net": {
        "senses": {
            "s1": "A mesh material for catching or containing.",
            "s2": "Remaining after all deductions (financial).",
            "s3": "The internet (informal)."
        },
        "domains": ["General", "Finance", "Technology"],
        "examples": {
            "s1": ["The fisherman cast his net into the sea.", "A safety net protects performers."],
            "s2": ["The net profit was $2 million.", "What's your net worth?"],
            "s3": ["I found this information on the net.", "She surfs the net for hours."]
        }
    },
    "pitch": {
        "senses": {
            "s1": "A throw in sports like baseball.",
            "s2": "The frequency of a sound.",
            "s3": "A sales presentation.",
            "s4": "A black sticky substance."
        },
        "domains": ["Sports", "Music", "Business"],
        "examples": {
            "s1": ["The pitcher threw a perfect pitch.", "That was a wild pitch."],
            "s2": ["Her voice hit a high pitch.", "Tune the guitar to the correct pitch."],
            "s3": ["He made his sales pitch to the investors.", "The pitch was convincing."],
            "s4": ["The deck was sealed with pitch.", "Pitch is derived from petroleum."]
        }
    },
    "plant": {
        "senses": {
            "s1": "A living organism that grows in the ground.",
            "s2": "An industrial facility or factory.",
            "s3": "To place something secretly."
        },
        "domains": ["Biology", "Industry", "Espionage"],
        "examples": {
            "s1": ["The plant needs more sunlight.", "She waters her plants every morning."],
            "s2": ["The manufacturing plant employs 500 workers.", "They toured the assembly plant."],
            "s3": ["Someone planted evidence at the scene.", "The spy was a plant."]
        }
    },
    "record": {
        "senses": {
            "s1": "A document containing information.",
            "s2": "The best performance ever achieved.",
            "s3": "A vinyl disc for music playback."
        },
        "domains": ["Administration", "Sports", "Music"],
        "examples": {
            "s1": ["Keep a record of all transactions.", "Medical records are confidential."],
            "s2": ["She broke the world record.", "That's a new personal record."],
            "s3": ["He collects vintage vinyl records.", "Put on a record."]
        }
    },
    "ring": {
        "senses": {
            "s1": "Circular jewelry worn on the finger.",
            "s2": "The sound of a bell or phone.",
            "s3": "A group of people working together, often criminally."
        },
        "domains": ["Fashion", "Communication", "Crime"],
        "examples": {
            "s1": ["He proposed with a diamond ring.", "The ring was passed down generations."],
            "s2": ["The phone's ring startled me.", "Give me a ring later."],
            "s3": ["Police busted a drug ring.", "A ring of thieves was arrested."]
        }
    },
    "rock": {
        "senses": {
            "s1": "A solid mineral substance.",
            "s2": "A genre of popular music.",
            "s3": "To move back and forth."
        },
        "domains": ["Geology", "Music"],
        "examples": {
            "s1": ["The hikers climbed over the rocks.", "The rock was smooth from erosion."],
            "s2": ["Rock music dominated the 1980s.", "They played rock at the concert."],
            "s3": ["The boat rocked in the waves.", "Don't rock the boat."]
        }
    },
    "run": {
        "senses": {
            "s1": "To move quickly on foot.",
            "s2": "To operate or manage.",
            "s3": "A continuous sequence.",
            "s4": "To stand as a candidate."
        },
        "domains": ["Physical", "Business", "Politics"],
        "examples": {
            "s1": ["She runs five miles every morning.", "Run to the store quickly."],
            "s2": ["He runs a successful business.", "The software runs on Windows."],
            "s3": ["The show had a long run on Broadway.", "A run of bad luck."],
            "s4": ["She decided to run for president.", "He's running for office."]
        }
    },
    "scale": {
        "senses": {
            "s1": "A device for measuring weight.",
            "s2": "The ratio of size between a map and reality.",
            "s3": "A series of musical notes.",
            "s4": "The small plates covering fish."
        },
        "domains": ["Measurement", "Music", "Biology"],
        "examples": {
            "s1": ["Step on the scale to check your weight.", "The kitchen scale is broken."],
            "s2": ["The map is drawn to scale.", "The model is 1/10 scale."],
            "s3": ["Practice your scales every day.", "Play the C major scale."],
            "s4": ["Remove the scales before cooking the fish.", "Fish scales glittered in the light."]
        }
    },
    "seal": {
        "senses": {
            "s1": "A marine mammal.",
            "s2": "A device for closing securely.",
            "s3": "An official emblem or stamp."
        },
        "domains": ["Biology", "General", "Official"],
        "examples": {
            "s1": ["Seals sunbathe on the rocks.", "The seal dove into the water."],
            "s2": ["Make sure the seal is airtight.", "Break the seal to open."],
            "s3": ["The document bore the royal seal.", "The notary applied her seal."]
        }
    },
    "sentence": {
        "senses": {
            "s1": "A grammatical unit of words.",
            "s2": "The punishment assigned by a court."
        },
        "domains": ["Grammar", "Law"],
        "examples": {
            "s1": ["Write a complete sentence.", "This sentence has five words."],
            "s2": ["The judge handed down a harsh sentence.", "He's serving a life sentence."]
        }
    },
    "set": {
        "senses": {
            "s1": "A group of related items.",
            "s2": "To place or put something.",
            "s3": "The scenery for a theatrical production.",
            "s4": "A unit of play in tennis."
        },
        "domains": ["General", "Entertainment", "Sports"],
        "examples": {
            "s1": ["She bought a new set of dishes.", "A set of tools."],
            "s2": ["Set the table for dinner.", "Set the alarm for 7 AM."],
            "s3": ["The film set was elaborate.", "They're building the set now."],
            "s4": ["She won the first set 6-4.", "Match point in the final set."]
        }
    },
    "shot": {
        "senses": {
            "s1": "The firing of a gun.",
            "s2": "A small serving of alcohol.",
            "s3": "A photograph.",
            "s4": "A medical injection."
        },
        "domains": ["Military", "Drinking", "Photography", "Medical"],
        "examples": {
            "s1": ["A shot rang out in the night.", "He's a good shot with a rifle."],
            "s2": ["Let's do a shot of tequila.", "Shots are on the house."],
            "s3": ["That's a great shot of the sunset.", "Get a close-up shot."],
            "s4": ["The nurse gave me a flu shot.", "The shot hurt a little."]
        }
    },
    "spring": {
        "senses": {
            "s1": "The season between winter and summer.",
            "s2": "A coiled metal device.",
            "s3": "A natural source of water.",
            "s4": "To jump or leap suddenly."
        },
        "domains": ["Time", "Mechanics", "Geography"],
        "examples": {
            "s1": ["Flowers bloom in spring.", "Spring break is next week."],
            "s2": ["The spring in the mattress is broken.", "Replace the spring mechanism."],
            "s3": ["The spring water is crystal clear.", "A natural hot spring."],
            "s4": ["The cat sprang onto the counter.", "He sprang to his feet."]
        }
    },
    "staff": {
        "senses": {
            "s1": "The employees of an organization.",
            "s2": "A long stick used for support or authority.",
            "s3": "The lines on which music is written."
        },
        "domains": ["Business", "Authority", "Music"],
        "examples": {
            "s1": ["The staff meeting is at 3 PM.", "They hired additional staff."],
            "s2": ["The wizard carried a wooden staff.", "The shepherd's staff."],
            "s3": ["Write the notes on the staff.", "The treble staff has five lines."]
        }
    },
    "stake": {
        "senses": {
            "s1": "A pointed wooden or metal post.",
            "s2": "An interest or share in something.",
            "s3": "Money risked in gambling."
        },
        "domains": ["General", "Finance", "Gambling"],
        "examples": {
            "s1": ["Drive the stake into the ground.", "Tie the plant to a stake."],
            "s2": ["She has a stake in the company.", "What's at stake here?"],
            "s3": ["The stakes are too high.", "He raised the stakes."]
        }
    },
    "stock": {
        "senses": {
            "s1": "Goods held for sale.",
            "s2": "Shares of a company's ownership.",
            "s3": "A liquid used as a base for soup.",
            "s4": "The handle of a rifle."
        },
        "domains": ["Retail", "Finance", "Cooking"],
        "examples": {
            "s1": ["Check if we have it in stock.", "The store is out of stock."],
            "s2": ["Tech stocks are performing well.", "She bought stock in Apple."],
            "s3": ["The soup is made with chicken stock.", "Simmer the stock for hours."],
            "s4": ["The rifle stock was walnut.", "Hold the stock firmly."]
        }
    },
    "stream": {
        "senses": {
            "s1": "A small natural body of flowing water.",
            "s2": "To transmit or receive digital content online."
        },
        "domains": ["Geography", "Technology"],
        "examples": {
            "s1": ["We crossed the stream on stepping stones.", "The stream flowed down the mountain."],
            "s2": ["I'll stream the concert live.", "What are you streaming tonight?"]
        }
    },
    "suit": {
        "senses": {
            "s1": "A matching jacket and trousers.",
            "s2": "A legal action or proceeding.",
            "s3": "A set of playing cards."
        },
        "domains": ["Fashion", "Law", "Games"],
        "examples": {
            "s1": ["He wore a tailored suit to the interview.", "The suit fit perfectly."],
            "s2": ["They filed a suit against the company.", "The lawsuit was dismissed."],
            "s3": ["Hearts is a suit in a deck of cards.", "Follow suit if you can."]
        }
    },
    "tank": {
        "senses": {
            "s1": "A large container for liquids or gas.",
            "s2": "An armored military vehicle.",
            "s3": "To fail completely (slang)."
        },
        "domains": ["General", "Military", "Slang"],
        "examples": {
            "s1": ["Fill up the gas tank.", "The fish tank needs cleaning."],
            "s2": ["The tank rolled across the battlefield.", "Tanks are heavily armored."],
            "s3": ["The movie tanked at the box office.", "Their stock price tanked."]
        }
    },
    "tender": {
        "senses": {
            "s1": "Soft or gentle; showing care.",
            "s2": "A formal offer to supply or buy.",
            "s3": "Legal currency for payment."
        },
        "domains": ["Personal", "Business", "Finance"],
        "examples": {
            "s1": ["The steak was perfectly tender.", "A tender moment between them."],
            "s2": ["The company submitted a tender for the project.", "We won the tender."],
            "s3": ["Cash is legal tender.", "The coins are no longer legal tender."]
        }
    },
    "terminal": {
        "senses": {
            "s1": "A building at an airport or bus station.",
            "s2": "A device for computer input/output.",
            "s3": "Relating to death; fatal."
        },
        "domains": ["Transportation", "Technology", "Medical"],
        "examples": {
            "s1": ["Meet me at Terminal B.", "The terminal was crowded."],
            "s2": ["Open a new terminal window.", "The command line terminal."],
            "s3": ["He was diagnosed with terminal cancer.", "Terminal illness."]
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
            "s1": ["I'll have toast for breakfast.", "Butter the toast."],
            "s2": ["Let's raise a toast to the newlyweds.", "They gave a toast at dinner."],
            "s3": ["If the boss finds out, we're toast.", "You're so toast."]
        }
    },
    "trunk": {
        "senses": {
            "s1": "The main stem of a tree.",
            "s2": "A large storage box.",
            "s3": "The rear luggage compartment of a car.",
            "s4": "An elephant's elongated nose."
        },
        "domains": ["Botany", "Storage", "Automotive", "Biology"],
        "examples": {
            "s1": ["The tree trunk was massive.", "Carve initials into the trunk."],
            "s2": ["Store old clothes in the trunk.", "A steamer trunk."],
            "s3": ["Put the groceries in the trunk.", "Pop the trunk open."],
            "s4": ["The elephant used its trunk to drink.", "The trunk can lift heavy loads."]
        }
    },
    "type": {
        "senses": {
            "s1": "A category or class of things.",
            "s2": "To write using a keyboard."
        },
        "domains": ["General", "Technology"],
        "examples": {
            "s1": ["What type of music do you like?", "This type of behavior is unacceptable."],
            "s2": ["Type your password and press enter.", "She can type 80 words per minute."]
        }
    },
    "vessel": {
        "senses": {
            "s1": "A ship or large boat.",
            "s2": "A container for liquids.",
            "s3": "A tube carrying blood in the body."
        },
        "domains": ["Maritime", "General", "Medical"],
        "examples": {
            "s1": ["The vessel sailed at dawn.", "A cargo vessel."],
            "s2": ["Pour water into the vessel.", "An earthen vessel."],
            "s3": ["Blood vessels carry oxygen.", "A burst blood vessel."]
        }
    },
    "volume": {
        "senses": {
            "s1": "The amount of space occupied.",
            "s2": "The loudness of sound.",
            "s3": "A single book in a series."
        },
        "domains": ["Physics", "Audio", "Publishing"],
        "examples": {
            "s1": ["Calculate the volume of the cylinder.", "Volume is measured in liters."],
            "s2": ["Turn down the volume.", "The volume was too loud."],
            "s3": ["I'm reading volume three of the series.", "The complete works in five volumes."]
        }
    },
    "wave": {
        "senses": {
            "s1": "A moving ridge on the surface of water.",
            "s2": "A gesture of greeting with the hand.",
            "s3": "A surge or increase in something."
        },
        "domains": ["Ocean", "Social", "General"],
        "examples": {
            "s1": ["The waves crashed against the shore.", "Surfers ride the waves."],
            "s2": ["She gave a friendly wave.", "Wave goodbye."],
            "s3": ["A new wave of COVID cases.", "A heat wave."]
        }
    },
    "well": {
        "senses": {
            "s1": "In good health or satisfactory manner.",
            "s2": "A deep hole from which water is drawn.",
            "s3": "An interjection indicating pause."
        },
        "domains": ["Health", "Geography"],
        "examples": {
            "s1": ["I hope you're feeling well.", "All is well."],
            "s2": ["Draw water from the well.", "The well ran dry."],
            "s3": ["Well, let me think about that.", "Well, that's surprising."]
        }
    },
    "wire": {
        "senses": {
            "s1": "A thin metal thread or strand.",
            "s2": "To transfer money electronically.",
            "s3": "A hidden microphone for surveillance."
        },
        "domains": ["Materials", "Finance", "Espionage"],
        "examples": {
            "s1": ["Connect the wires carefully.", "Barbed wire fenced the property."],
            "s2": ["Wire the money to my account.", "The funds were wired immediately."],
            "s3": ["The informant wore a wire.", "They bugged him with a wire."]
        }
    },
    "yield": {
        "senses": {
            "s1": "To produce or provide.",
            "s2": "To give way or surrender.",
            "s3": "The return on an investment."
        },
        "domains": ["Agriculture", "General", "Finance"],
        "examples": {
            "s1": ["The farm yields 100 tons annually.", "The experiment yielded results."],
            "s2": ["Yield to oncoming traffic.", "He yielded to their demands."],
            "s3": ["The bond yield is 4.5%.", "High-yield investments are risky."]
        }
    }
}

# Additional high-frequency polysemes
ADDITIONAL_WORDS = {
    "address": {
        "senses": {
            "s1": "A location where someone lives or works.",
            "s2": "To speak to or handle a matter.",
            "s3": "A formal speech."
        },
        "domains": ["Location", "Communication"],
        "examples": {
            "s1": ["What's your email address?", "The package was sent to the wrong address."],
            "s2": ["We need to address this issue.", "The report addresses several concerns."],
            "s3": ["The president's address was broadcast nationwide.", "A keynote address."]
        }
    },
    "case": {
        "senses": {
            "s1": "A container or covering.",
            "s2": "A legal action or lawsuit.",
            "s3": "An instance or example."
        },
        "domains": ["Storage", "Law", "General"],
        "examples": {
            "s1": ["Put the guitar in its case.", "A briefcase."],
            "s2": ["The case was dismissed.", "They built a strong case."],
            "s3": ["In that case, let's proceed.", "A case of mistaken identity."]
        }
    },
    "check": {
        "senses": {
            "s1": "To examine or verify.",
            "s2": "A written order for payment.",
            "s3": "A pattern of squares.",
            "s4": "An attack on the king in chess."
        },
        "domains": ["General", "Finance", "Games"],
        "examples": {
            "s1": ["Check your work before submitting.", "I'll check the schedule."],
            "s2": ["Pay by check or cash.", "The check bounced."],
            "s3": ["She wore a check pattern dress.", "A red and white check."],
            "s4": ["The bishop put the king in check.", "Checkmate!"]
        }
    },
    "cool": {
        "senses": {
            "s1": "Moderately cold temperature.",
            "s2": "Calm and composed.",
            "s3": "Fashionable or impressive (slang)."
        },
        "domains": ["Temperature", "Emotion", "Slang"],
        "examples": {
            "s1": ["The weather turned cool.", "Keep the drinks cool."],
            "s2": ["Stay cool under pressure.", "She remained cool during the crisis."],
            "s3": ["That car is so cool.", "Your new haircut is cool."]
        }
    },
    "expression": {
        "senses": {
            "s1": "A facial look showing emotion.",
            "s2": "A phrase or way of saying something.",
            "s3": "A mathematical formula."
        },
        "domains": ["Emotion", "Language", "Mathematics"],
        "examples": {
            "s1": ["His expression showed confusion.", "A pained expression."],
            "s2": ["That's just an expression.", "Freedom of expression."],
            "s3": ["Simplify the algebraic expression.", "Evaluate the expression."]
        }
    },
    "figure": {
        "senses": {
            "s1": "A number or statistic.",
            "s2": "A person's body shape.",
            "s3": "An important person.",
            "s4": "To understand or calculate."
        },
        "domains": ["Mathematics", "Physical", "Verb"],
        "examples": {
            "s1": ["The figures don't add up.", "Check the sales figures."],
            "s2": ["She has a slim figure.", "A shadowy figure."],
            "s3": ["A key figure in the movement.", "A public figure."],
            "s4": ["I can't figure out this puzzle.", "Let me figure it out."]
        }
    },
    "light": {
        "senses": {
            "s1": "Electromagnetic radiation visible to the eye.",
            "s2": "Not heavy in weight.",
            "s3": "To ignite or illuminate."
        },
        "domains": ["Physics", "Weight", "Action"],
        "examples": {
            "s1": ["The light was too bright.", "Natural light is best for photos."],
            "s2": ["This bag is very light.", "A light fabric."],
            "s3": ["Light the candles.", "Light up the room."]
        }
    },
    "minute": {
        "senses": {
            "s1": "A unit of time (60 seconds).",
            "s2": "Extremely small (pronounced my-NOOT)."
        },
        "domains": ["Time", "Size"],
        "examples": {
            "s1": ["Wait a minute.", "The meeting lasted 30 minutes."],
            "s2": ["The particles are minute.", "Minute details."]
        }
    },
    "novel": {
        "senses": {
            "s1": "A long fictional narrative book.",
            "s2": "New and original."
        },
        "domains": ["Literature", "Innovation"],
        "examples": {
            "s1": ["She's writing her first novel.", "A bestselling novel."],
            "s2": ["That's a novel approach.", "A novel idea."]
        }
    },
    "objective": {
        "senses": {
            "s1": "A goal or aim.",
            "s2": "Not influenced by personal feelings; impartial."
        },
        "domains": ["Planning", "Judgment"],
        "examples": {
            "s1": ["Our main objective is growth.", "Mission objective."],
            "s2": ["Give an objective assessment.", "Remain objective."]
        }
    },
    "order": {
        "senses": {
            "s1": "A request for goods or services.",
            "s2": "A methodical arrangement.",
            "s3": "A command or instruction.",
            "s4": "A religious community."
        },
        "domains": ["Commerce", "Organization", "Authority"],
        "examples": {
            "s1": ["I placed an order online.", "Your order will arrive tomorrow."],
            "s2": ["Put the files in alphabetical order.", "Law and order."],
            "s3": ["The general gave the order to attack.", "Follow orders."],
            "s4": ["A monastic order.", "The Order of Knights."]
        }
    },
    "party": {
        "senses": {
            "s1": "A social gathering.",
            "s2": "A political organization.",
            "s3": "A person or group in a legal matter."
        },
        "domains": ["Social", "Politics", "Law"],
        "examples": {
            "s1": ["We're throwing a birthday party.", "The party was fun."],
            "s2": ["Which party do you support?", "The Democratic Party."],
            "s3": ["The third party involved.", "Both parties agreed to settle."]
        }
    },
    "point": {
        "senses": {
            "s1": "A sharp tip.",
            "s2": "A unit of scoring.",
            "s3": "An argument or idea.",
            "s4": "A specific location."
        },
        "domains": ["Physical", "Sports", "Discussion", "Geography"],
        "examples": {
            "s1": ["The point of the needle is sharp.", "Pencil point."],
            "s2": ["They scored 3 points.", "Point guard."],
            "s3": ["That's a good point.", "Get to the point."],
            "s4": ["Meeting point.", "The point of no return."]
        }
    },
    "present": {
        "senses": {
            "s1": "The current time.",
            "s2": "A gift.",
            "s3": "To show or introduce."
        },
        "domains": ["Time", "Social", "Communication"],
        "examples": {
            "s1": ["Live in the present.", "The present moment."],
            "s2": ["Open your birthday present.", "A present for you."],
            "s3": ["Present the findings.", "May I present the speaker."]
        }
    },
    "project": {
        "senses": {
            "s1": "A planned undertaking.",
            "s2": "To estimate or forecast.",
            "s3": "To throw or cast forward."
        },
        "domains": ["Work", "Finance", "Physical"],
        "examples": {
            "s1": ["Start a new project.", "The construction project."],
            "s2": ["Project next year's revenue.", "Projected growth."],
            "s3": ["Project your voice.", "Project an image on the screen."]
        }
    },
    "record": {
        "senses": {
            "s1": "Information stored for reference.",
            "s2": "A best performance ever achieved.",
            "s3": "A vinyl disc for music."
        },
        "domains": ["Documentation", "Sports", "Music"],
        "examples": {
            "s1": ["Keep a record of expenses.", "Medical records."],
            "s2": ["She broke the world record.", "A new record time."],
            "s3": ["Play a vinyl record.", "Record collector."]
        }
    },
    "second": {
        "senses": {
            "s1": "A unit of time.",
            "s2": "Coming after the first.",
            "s3": "To support a motion."
        },
        "domains": ["Time", "Order", "Parliamentary"],
        "examples": {
            "s1": ["Wait a second.", "60 seconds in a minute."],
            "s2": ["Finish second place.", "The second chapter."],
            "s3": ["I second that motion.", "Seconded!"]
        }
    },
    "sign": {
        "senses": {
            "s1": "A notice or indicator.",
            "s2": "To write one's signature.",
            "s3": "A gesture in sign language."
        },
        "domains": ["Communication", "Legal", "Language"],
        "examples": {
            "s1": ["Follow the road signs.", "A good sign."],
            "s2": ["Sign the contract.", "Sign here please."],
            "s3": ["She learned to sign.", "Sign language."]
        }
    },
    "state": {
        "senses": {
            "s1": "A condition or status.",
            "s2": "A political territory.",
            "s3": "To express in words."
        },
        "domains": ["Condition", "Government", "Communication"],
        "examples": {
            "s1": ["The house is in poor state.", "State of mind."],
            "s2": ["The state of California.", "Head of state."],
            "s3": ["State your name.", "As stated earlier."]
        }
    },
    "subject": {
        "senses": {
            "s1": "A topic or theme.",
            "s2": "A citizen under a monarch.",
            "s3": "The grammatical subject of a sentence.",
            "s4": "To cause to undergo."
        },
        "domains": ["Education", "Government", "Grammar", "Action"],
        "examples": {
            "s1": ["Change the subject.", "My favorite subject is history."],
            "s2": ["A subject of the Queen.", "Royal subjects."],
            "s3": ["The subject of the sentence.", "Subject-verb agreement."],
            "s4": ["Subject to approval.", "Subjected to testing."]
        }
    },
    "table": {
        "senses": {
            "s1": "A piece of furniture.",
            "s2": "A grid of data.",
            "s3": "To postpone discussion."
        },
        "domains": ["Furniture", "Data", "Procedural"],
        "examples": {
            "s1": ["Set the table for dinner.", "A wooden table."],
            "s2": ["Refer to table 3.", "A table of contents."],
            "s3": ["Let's table this discussion.", "The motion was tabled."]
        }
    },
    "term": {
        "senses": {
            "s1": "A word or phrase.",
            "s2": "A period of time.",
            "s3": "Conditions or stipulations."
        },
        "domains": ["Language", "Time", "Legal"],
        "examples": {
            "s1": ["Define the term.", "Technical terms."],
            "s2": ["A three-year term.", "Term limits."],
            "s3": ["Terms and conditions.", "On good terms."]
        }
    },
    "title": {
        "senses": {
            "s1": "The name of a work.",
            "s2": "A formal designation.",
            "s3": "Legal ownership."
        },
        "domains": ["Media", "Formal", "Legal"],
        "examples": {
            "s1": ["The book's title is intriguing.", "Title page."],
            "s2": ["Her title is Director.", "A royal title."],
            "s3": ["The title deed.", "Clear title."]
        }
    },
    "track": {
        "senses": {
            "s1": "A path or trail.",
            "s2": "A song on an album.",
            "s3": "To follow or monitor."
        },
        "domains": ["Path", "Music", "Surveillance"],
        "examples": {
            "s1": ["Stay on the track.", "Railroad track."],
            "s2": ["Play the next track.", "My favorite track."],
            "s3": ["Track the package.", "Tracking your progress."]
        }
    },
    "train": {
        "senses": {
            "s1": "A rail vehicle.",
            "s2": "To teach or prepare.",
            "s3": "A trailing part of a garment."
        },
        "domains": ["Transportation", "Education", "Fashion"],
        "examples": {
            "s1": ["Catch the train.", "The train was delayed."],
            "s2": ["Train the new employees.", "Train for the marathon."],
            "s3": ["The wedding dress had a long train.", "A bridal train."]
        }
    },
    "watch": {
        "senses": {
            "s1": "A timepiece worn on the wrist.",
            "s2": "To observe or look at.",
            "s3": "A period of guard duty."
        },
        "domains": ["Accessories", "Observation", "Security"],
        "examples": {
            "s1": ["Check your watch.", "A smartwatch."],
            "s2": ["Watch the movie.", "Watch out!"],
            "s3": ["The night watch.", "Keep watch."]
        }
    }
}

# Merge catalogs
WORD_CATALOG.update(ADDITIONAL_WORDS)


def generate_bundle(word: str, target_sense: str, difficulty: float = 0.5) -> dict:
    """Generate a single polysemy bundle for a word and target sense."""
    if word not in WORD_CATALOG:
        return None

    catalog = WORD_CATALOG[word]
    senses = catalog["senses"]
    examples = catalog["examples"]
    domains = catalog.get("domains", ["General"])

    if target_sense not in senses:
        return None

    # Get target sense examples
    target_examples = examples.get(target_sense, [])
    if len(target_examples) < 2:
        return None

    # Get negative sense (different from target)
    other_senses = [s for s in senses.keys() if s != target_sense]
    if not other_senses:
        return None

    negative_sense = random.choice(other_senses)
    negative_examples = examples.get(negative_sense, [])
    if not negative_examples:
        return None

    # Create bundle
    bundle_id = f"PB_{word.upper()}_{target_sense.upper()}_{random.randint(100, 999)}"

    anchor = random.choice(target_examples)
    positive = [ex for ex in target_examples if ex != anchor]
    positive_text = positive[0] if positive else target_examples[0]

    negative_text = random.choice(negative_examples)

    # Hard negative: Uses the word in negative sense but with lexical overlap
    hard_negative = negative_text  # For now, use same as negative
    # In production, we'd craft more sophisticated hard negatives

    bundle = {
        "bundle_id": bundle_id,
        "word": word,
        "sense_catalog": senses,
        "metadata": {
            "domains": domains,
            "difficulty_score": difficulty,
            "polysemy_count": len(senses),
            "target_sense": target_sense
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

    for word, catalog in WORD_CATALOG.items():
        senses = list(catalog["senses"].keys())

        for _ in range(bundles_per_word):
            # Randomly select target sense
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
    print("SemanticPhase v2 Bundle Generator")
    print("=" * 50)
    print(f"Words in catalog: {len(WORD_CATALOG)}")

    # Generate bundles
    random.seed(42)
    bundles = generate_all_bundles(bundles_per_word=4)

    print(f"Generated {len(bundles)} bundles")

    # Save to output
    output_path = Path("paradigm_factory/output/semanticphase_v2_bundles.jsonl")
    save_bundles(bundles, output_path)

    # Show sample
    print("\nSample bundle:")
    print(json.dumps(bundles[0], indent=2))

    return bundles


if __name__ == "__main__":
    main()
