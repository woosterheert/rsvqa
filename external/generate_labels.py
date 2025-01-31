CLC_hierarchy = {'artificial areas':
                 {'urban fabric':['continuous urban fabric', 'discontinuous urban fabric'],
                  'industrial, commercial, and transport units':['industrial or commercial units', 'road and rail networks and associated land', 'port areas', 'airports'],
                  'mine, dump and construction sites':['mineral extraction sites', 'dump sites', 'construction sites'],
                  'artificial non-agricultural vegetated areas':['green urban areas', 'sport and leisure facilities']
                  },
                 'agricultural areas':
                 {'arable land':['non-irrigated arable land', 'permanently irrigated land', 'rice fields'],
                  'permanent crops':['vineyards', 'fruit trees and berry plantations', 'olive groves'],
                  'pastures':['pastures'],
                  'heterogeneous agricultural areas':['annual crops associated with permanent crops', 'complex cultivation patterns', 'land principally occupied by agriculture, with significant areas of natural vegetation', 'agro-forestry areas']
                  },
                 'forest and seminatural areas':
                 {'forests':['broad-leaved forest', 'coniferous forest', 'mixed forest'],
                  'scrub and/or herbaceous vegetation associations':['natural grassland', 'moors and heathland', 'sclerophyllous vegetation', 'transitional woodland/shrub'],
                  'open spaces with little or no vegetation':['beaches, dunes, sands', 'bare rock', 'sparsely vegetated areas', 'burnt areas']#, 'glaciers and perpetual snow']
                  },
                 'wetlands':
                 {'inland wetlands':['inland marshes', 'peatbogs'],
                  'maritime wetlands':['salt marshes', 'salines', 'intertidal flats']
                  },
                 'water bodies':
                 {'inland waters':['water courses', 'water bodies'],
                  'marine waters':['coastal lagoons', 'estuaries', 'sea and ocean']
                  }
                }


CLC_dict = {'mixed forest':['mixed forest', 'forest'],
            'coniferous forest':['coniferous forest', 'forest'],
       'non-irrigated arable land':['non-irrigated arable area', 'arable area', 'agricultural area'],
       'transitional woodland/shrub':['woodland', 'shrub', 'scrub'],
       'broad-leaved forest':['broad-leaved forest', 'forest'],
       'land principally occupied by agriculture, with significant areas of natural vegetation':['agricultural area'],
       'complex cultivation patterns':['agricultural area'],
       'pastures':['pastures'],
       'water bodies':['water body', 'water area', 'inland water', 'water'],
       'sea and ocean':['sea', 'ocean', 'marine water', 'water'],
       'discontinuous urban fabric':['discontinuous urban fabric', 'urban fabric', 'artificial area'],
       'agro-forestry areas':['agro-forestry area', 'agricultural area'],
       'peatbogs':['peat bog', 'inland wetland', 'wetland'],
       'permanently irrigated land':['permanently irrigated area', 'arable area', 'agricultural area'],
       'industrial or commercial units':['industrial area', 'commercial area', 'artificial area'],
       'natural grassland':['natural grassland', 'grassland', 'scrub'],
       'olive groves':['olive groves', 'permanent crop', 'agricultural area'],
       'sclerophyllous vegetation':['sclerophyllous vegetation', 'vegetation', 'scrub'],
       'continuous urban fabric':['continuous urban fabric', 'urban fabric', 'artificial area'],
       'water courses':['water courses', 'inland water', 'water'],
       'vineyards':['vineyards', 'permanent crops', 'agricultural area'],
       'annual crops associated with permanent crops':['annual crops', 'agricultural area'],
       'inland marshes':['inland marshes', 'inland wetland', 'wetland'],
       'moors and heathland':['moors', 'heathland', 'scrub'],
       'sport and leisure facilities':['leisure facilities', 'artificial area'],
       'fruit trees and berry plantations':['orchard', 'permanent crop', 'agricultural area'],
       'mineral extraction sites':['mine', 'artificial area'],
       'rice fields':['rice fields', 'arable area', 'agricultural area'],
       'road and rail networks and associated land':['road', 'railway', 'transport unit', 'artificial area'],
       'bare rock':['rock'],
       'green urban areas':['green urban area', 'artificial vegetated area', 'artificial area'],
       'beaches, dunes, sands':['beach'],
       'sparsely vegetated areas':['sparsely vegetated area'],
       'salt marshes':['salt marshes', 'maritime wetland', 'wetland'],
       'coastal lagoons':['lagoon', 'marine water', 'water'],
       'construction sites':['construction site','artificial area'],
       'estuaries':['estuary', 'marine water', 'water'],
       'intertidal flats':['tidal', 'maritime wetland', 'wetland'],
       'airports':['airport', 'transport unit', 'artificial area'],
       'dump sites':['dump', 'artificial area'],
       'port areas':['port', 'artificial area'],
       'salines':['salt pond', 'maritime wetland', 'wetland'],
       'burnt areas':['burnt area']}

import random

def get_labels(CLC_labels):
  roots = CLC_hierarchy.keys()

  L1_present = []
  L1_absent = []
  L2_present = []
  L2_absent = []
  L3_present = []
  L3_absent = []

  for L1 in roots:
    L1_present_b = False
    for L2 in CLC_hierarchy[L1].keys():
      L2_present_b = False
      for L3 in CLC_hierarchy[L1][L2]:
        if L3 in CLC_labels:
          L3_present.append((L1, L2, L3))
          L2_present_b = True
          L1_present_b = True
        else:
          L3_absent.append([L1, L2, L3])
      if L2_present_b:
        L2_present.append([L1, L2])
      else:
        L2_absent.append([L1, L2])
    if L1_present_b:
      L1_present.append([L1])
    else:
      L1_absent.append([L1])

  output = {"L1":{"present":L1_present, "absent":L1_absent}, "L2":{"present":L2_present, "absent":L2_absent}, "L3":{"present":L3_present, "absent":L3_absent}}

  return output

def generate_yes_no_question(labels):
  #Choose CLC hierarchical level for the question
  question_level = random.choice(["L1","L2","L3"])
  labels_present = labels[question_level]["present"]
  labels_absent = labels[question_level]["absent"]
  question = None

  #Choose the answer to the question
  dice = random.random()
  if dice < 0.5:
    answer = "yes"
  else:
    answer = "no"

  if answer == "no":
    dice = random.random()
    if dice < 0.5:
      if len(labels_absent) >= 1:
        LC = random.choice(labels_absent)[-1]
        question = ("presence", "logical 1", "#", [LC], "no")
    elif dice < 0.65:
      if len(labels_absent) >= 1 and len(labels_present) >= 1:
        LCs = [random.choice(labels_absent)[-1], random.choice(labels_present)[-1]]
        random.shuffle(LCs)
        question = ("presence", "logical 2", "# and #", LCs, "no")
    elif dice < 0.725:
      if len(labels_absent) >= 2:
        random.shuffle(labels_absent)
        LCs = [labels_absent[0][-1], labels_absent[1][-1]]  
        random.shuffle(LCs)
        question = ("presence", "logical 2", "# and #", LCs, "no")
    elif dice < 0.8:
      if len(labels_absent) >= 2:
        random.shuffle(labels_absent)
        LCs = [labels_absent[0][-1], labels_absent[1][-1]]
        question = ("presence", "logical 2", "# or #", LCs, "no")
    else:
      if len(labels_absent) >= 2 and len(labels_present) >= 1:
        random.shuffle(labels_absent)
        neg = [labels_absent[0][-1], labels_absent[1][-1]]
        pos = random.choice(labels_present)[-1]
        dice2 = random.random()
        if dice2 < 0.5:
          LCs = neg + [pos]
          question = ("presence", "logical 3", "# or # and #", LCs, "no")
        else:
          LCs = [pos] + neg
          question = ("presence", "logical 3", "# and # or #", LCs, "no")



  if answer == "yes":
    #Choose the number of positive samples
    dice = random.random()
    if dice < 0.5:
      if len(labels_present) >= 1:
        LC = random.choice(labels_present)[-1]
        question = ("presence", "logical 1", "#", [LC], "yes")
    elif dice < 0.65:
      if len(labels_present) >= 2:
        random.shuffle(labels_present)
        LCs = [labels_present[0][-1], labels_present[1][-1]]
        question = ("presence", "logical 2", "# and #", LCs, "yes")
    elif dice < 0.8:
      if len(labels_present) >= 1 and len(labels_absent) >= 1:
        LCs = [random.choice(labels_present)[-1], random.choice(labels_absent)[-1]]
        random.shuffle(LCs)
        question = ("presence", "logical 2", "# or #", LCs, "yes")
    else:
      if len(labels_present) >= 2 and len(labels_absent) >= 1:
        random.shuffle(labels_present)
        LC1 = labels_present[0][-1]
        LC2 = labels_present[1][-1]
        LC3 = random.choice(labels_absent)[-1]
        dice2 = random.random()
        pos = [LC1, LC2]
        random.shuffle(pos)
        if dice2 < .25:
          LCs = pos + [LC3]
          question = ("presence", "logical 3", "# and # or #", LCs, "yes")
        elif dice2 < .5:
          LCs = [LC3] + pos
          question = ("presence", "logical 3", "# or # and #", LCs, "yes")
        elif dice2 < .75:
          LCs = pos + [LC3]
          question = ("presence", "logical 3", "# or # and #", LCs, "yes")
        else:
          LCs = [pos[0], LC3, pos[1]]
          question = ("presence", "logical 3", "# and # or #", LCs, "yes")
  
  #Check that there is no LC containing "and"
  if question:
    for LC in question[3]:
      if "and/or" in LC or " or " in LC or " and " in LC:
        question = None
  return question


def generate_LC_question(labels):
  question = None
  question_level = random.choices(["L1", "L2", "L3", "L1L2L3"], cum_weights=[40, 50, 90, 100])[0]
  if question_level != "L1L2L3":
    labels_present = labels[question_level]["present"]
    
    dice = random.random()
    if dice < 0.75:
      LCs = []
      for label in labels_present:
        LCs += [label[-1]]
      question = ("LC", "type 1", question_level, sorted(LCs))
    elif dice < 0.9:
      if len(labels_present) >= 1:
        LCs = []
        LC1 = random.choice(labels_present)[-1]
        for label in labels_present:
          if label[-1] != LC1:
            LCs += [label[-1]]
        question = ("LC", "type 2", question_level, [LC1] + sorted(LCs))
    else:
      if len(labels_present) >= 2:
        LCs = []
        random.shuffle(labels_present)
        LC1 = labels_present[0][-1]
        LC2 = labels_present[1][-1]
        for label in labels_present:
          if label[-1] != LC1 and label[-1] != LC2:
            LCs += [label[-1]]
        question = ("LC", "type 3", question_level, [LC1, LC2] + sorted(LCs))

  else:
    labels_present = []
    for lvl in ["L1", "L2", "L3"]:
      for label in labels[lvl]["present"]:
        if label[-1] not in labels_present:
          labels_present += [label[-1]]
    question = ("LC", "type 1", question_level, labels_present)
  #Check that there is no label with "and"
  if question:
    for label in question[3]:
      if "and/or" in label or " or " in label or " and " in label:
        question = None
  return question
  

def get_text_representation_of_question(question):
  outputq = ""
  outputa = ""
  if question[0] == "presence":
    outputa = question[4]
    if question[1] == "logical 1":
      outputq = random.choice(["Are there some #", "Are some # present"])
      outputq = outputq.replace("#", question[3][0])
    elif question[1] == "logical 2":
      outputq = random.choice(["Are there #", "Are # present"])
      outputq = outputq.replace("#", question[2])
      outputq = outputq.replace("#", question[3][0], 1)
      outputq = outputq.replace("#", question[3][1])
    elif question[1] == "logical 3":
      outputq = random.choice(["Are there #", "Are # present"])
      outputq = outputq.replace("#", question[2])
      for i in range(3):
        outputq = outputq.replace("#", question[3][i], 1)
    dice = random.random()
    if dice < .25:
      outputq += " in the image?"
    elif dice < .5:
      outputq += " in the scene?"
    else:
      outputq += "?"
  else:
    answer_LCs = []
    if question[1] == "type 1":
      answer_LCs = question[3]
      if question[2] != "L1L2L3":
        outputq = random.choice(["What " + question[2] + " land cover classes are there", "Which " + question[2] + " land cover classes are", "Which " + question[2] + " classes are"])
      else:
        outputq = random.choice(["What are all the land cover classes", "Which land cover classes are"])
    elif question[1] == "type 2":
      outputq = random.choice(["Besides #, what land cover classes are", "In addition to #, what are the land cover classes", "In addition to #, which classes are", "Besides #, which classes are"])
      outputq = outputq.replace("#", question[3][0])
      answer_LCs = question[3][1:]
    else:
      outputq = random.choice(["Besides # and #, what land cover classes are", "Besides # and #, what classes are", "In addition to # and #, what are the land cover classes", "In addition to # and #, what classes are"])
      outputq = outputq.replace("#", question[3][0], 1)
      outputq = outputq.replace("#", question[3][1], 1)
      answer_LCs = question[3][2:]
    if len(answer_LCs) == 0:
      outputa = "None"
    elif len(answer_LCs) == 1:
      outputa = answer_LCs[0]
    else:
      outputa = answer_LCs[0]
      for i in range(1, len(answer_LCs) - 1):
        outputa += ", " + answer_LCs[i]
      outputa += " and " + answer_LCs[-1]
    dice = random.random()
    if dice < .5:
      outputq += " in the image?"
    else:
      outputq += " in the scene?"
  
  return outputq, outputa


def ask_questions(CLC_labels, number=10, tries=1000, presence_only=False):
  generic_questions = []
  questions = []
  answers = []
  labels = get_labels([label.lower() for label in CLC_labels])
  while number and tries:
    tries -= 1
    #choose answer type:
    dice = random.random()

    if presence_only:
      question_type = "yes/no"
      question = generate_yes_no_question(labels)
    else:
      if dice < 0.5:
        question_type = "yes/no"
        question = generate_yes_no_question(labels)
      else:
        question_type = "LC"
        question = generate_LC_question(labels)
    
    if question and question not in generic_questions:
      generic_questions.append(question)
      txt_question, txt_answer = get_text_representation_of_question(question)
      questions.append(txt_question)
      answers.append(txt_answer)
      number-=1

  return questions, answers, generic_questions