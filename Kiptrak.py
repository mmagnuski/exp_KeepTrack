#!/usr/bin/env python
# -*- coding: utf-8 -*-

# QUESTIONs:
# [ ] możliwe jest uzyskanie takiej sekwencji ilości targetów:
#     [4, 4, 4, 2, 3, 3, 2, 4, 3, 3, 2, 2, 4, 2, 3]
#     alternatywa - mieszanie w grupach [2, 3, 4] np:
#     [3, 2, 4, 4, 3, 2, 3, 4, 2, 2, 3, 4, 3, 4, 2]
# TASKS:
# [ ] instrukcje? - prześle Natalia
# [ ] ? 'space' - do kończenia nagrywania dźwięku?
# [X] save RT and mic init time
# [X] jakie losowanie do triali treningowych? (zależne)
# [X] jakiś feedback w treningu? ("poprawna odpowiedź to:" bez dźwięku)
# 

# SETTINGS
# --------

# timings (in ms)
times = {}
times['categoriesFirst'] = 1500
times['fixation']        = 500
times['word']            = 1000
times['maxRespTime']     = 1000
times['feedback']        = 3500

# exp is a dictionary that holds
# information about the experiment
exp = {}
exp['times'] = times

# number of trials etc.:
# CHANGE
exp['trainingTrials']   = [2, 3, 4]
exp['numberOfTrials']   = 12
exp['categoriesHPos']   = -0.3
exp['activeButton']     = 'space'


# SOUND
# -----
THRESHOLD       = 500   # not sure if the threshold is adequate...
CHUNK_SIZE      = 1024
RATE            = 44100
SILENT_TRESHOLD = 87


# IMPORTS
# -------
print 'Importing os and sys...'
import os
import sys

print 'Importing numpy...'
import numpy as np
print 'Importing pandas...'
import pandas as pd
print 'Importing re...'
import re

# debug
print 'Python version running this script:\n', sys.version

# imports for voice recording
print 'Importing array and struct...'
from sys import byteorder
from array import array
from struct import pack

print 'Importing pyAudio...'
print '(if this fails - try running the procedure again)'
import pyaudio
print 'Importing wave...'
import wave
print 'Importing PsychoPy...'
from psychopy import core, visual, gui, event
print 'ignore the error above - do not worry :)'
print '\nall imports done.\n\n'


# DEFINITIONS
# -----------

# sound
# -----
FORMAT = pyaudio.paInt16

def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

def add_silence(snd_data, seconds):
    "Add silence to the end of 'snd_data' of length 'seconds' (float)"
    r = array('h', [0])
    r.extend(snd_data)
    r.extend([0 for i in xrange(int(seconds*RATE))])
    return r

def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    # snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def record(clock, time_list):
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.

    Normalizes the audio, trims silence from the 
    start and end, and pads with 0.5 seconds of 
    blank sound to make sure VLC et al can play 
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    # get initialization time
    time_list.append(clock.getTime())

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            time_list.append(clock.getTime())
            snd_started = True

        if snd_started and num_silent > SILENT_TRESHOLD:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    return sample_width, r

def record_to_file(path, clock):
    "Records from the microphone and outputs the resulting data to 'path'"
    time_list = []
    sample_width, data = record(clock, time_list)
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()
    return time_list

# Find N smallest (recursive)
def nSmallest(vec, N, use = None):
	'''
	ind = nSmallest(vec, N, use = None)

	returns indices of N smallest values
	from vector vec.
	if less than N items have the minimal value,
	items with next minimal value are added
	if at some point the numerosity of selected value
    indices is greater than N - highest-value items
    (out of those already selected) are chosen from 
    randomly so that overall N item indices are returned

    to use a subset of vec pass 'use' - a boolean vector
    with False at indices of items that should be omitted.

	'''

	# assumes vec is numpy array!
	vecLen = len(vec)
	if use == None:
		use = np.ones(vecLen, dtype = bool)

	# all indices
	allind = np.arange(vecLen)

	# select only those used:
	useind = allind[use]
	usevec = vec[use]

	# we assume it is numpy array, easier:
	ind = useind[ usevec == min(usevec) ]

	# now return by choosing from ind or
	# calling nSmallest again
	indLen = len(ind)

	if indLen == N:
		vec[ind] += 1
		return list(ind)		
	elif indLen > N:
		# select randomly
		np.random.shuffle(ind)
		ind = sorted( ind[:N] )
		vec[ind] += 1
		return list(ind)
	else:
		# not enough chosen
		use[ind] = False
		vec[ind] += 1
		diffN = N - indLen

		# recursive call
		ind = list(ind) + nSmallest(vec, diffN, use = use)
		ind.sort()
		return ind

def fillz(val, num):
    '''
    fillz(val, num)
    exadds zero to the beginning of val so that length of
    val is equal to num. val can be string, int or float
    '''
    
    # if not string - turn to a string
    if not isinstance(val, basestring):
        val = str(val)
     
    # add zeros
    ln = len(val)
    if ln < num:
        return '0' * (num - ln) + val
    else:
        return val

def ms2frames(times, frame_time):
    
	tp = type(times)
	if tp == type([]):
	    frms = []
	    for t in times:
	        frms.append( int( round(t / frame_time) ) )
	elif tp == type({}):
	    frms = {}
	    for t in times.keys():
	        frms[t] = int( round(times[t] / frame_time) )
	return frms

def free_filename(pth, subj, givenew = True):

	# list files within the dir
	fls = os.listdir(pth)
	n_undsc_in_subj = 0
	ind = -1
	while True:
		try:
			ind = subj['symb'][ind+1:-1].index('_')
			n_undsc_in_subj += 1
		except (IndexError, ValueError):
			break

	# predefined file pattern
	subject_pattern = subj['symb'] + '_' + r'[0-9]+'

	# check files for pattern
	reg = re.compile(subject_pattern)
	used_files = [reg.match(itm).group() for ind, itm in enumerate(fls) \
		if not (reg.match(itm) == None)]
	used_files_int = [int(f.split('_')[1 + n_undsc_in_subj][0:2]) for f in used_files]

	if givenew:
		if len(used_files) == 0:
			# if there are no such files - return prefix + S01
			return 1, subj['symb'] + '_01'
		else:
			# check max used number:
			mx = 1
			for temp in used_files_int:
				if temp > mx:
					mx = temp

			return mx + 1, subj['symb'] + '_' + fillz(mx + 1, 2)
	else:
		return not subj['ind'] in used_files_int


def generateTrials(subj, exp):

	# create empty dataframe
	# ----------------------

	# column names
	colNamesCat  = ['targetCategory' + fillz(x, 2) for x in range(1,5)]
	colNamesLast = ['lastWordCat' + fillz(x, 2) for x in range(1,5)]
	colNamesWord = ['word' + fillz(x, 2) for x in range(1,16)]
	colNamesTarg = ['wordIsTarget' + fillz(x, 2) for x in range(1,16)]

	# join column names
	colNames = ['ifExp', 'soundFile', 'micInitTime', 'RT', 'N'] + \
		colNamesCat + colNamesLast + colNamesWord + colNamesTarg

	# create DataFrame for results
	numTrain = 3
	df = pd.DataFrame(index=np.arange(0, numTrain + exp['numberOfTrials']), columns = colNames )


	# presented dict
	# --------------
	presented = {}
	presented['asTargetCategory'] = np.zeros(exp['catNum'],   dtype = int)
	presented['asTarget']         = np.zeros(exp['wordsNum'], dtype = int)
	presented['asNonTarget']      = np.zeros(exp['wordsNum'], dtype = int)
	presented['asLast']           = np.zeros(exp['wordsNum'], dtype = int)

	# ensure categ is numpy array
	exp['categ'] = np.array(exp['categ'])


	# CHOOSE CATEGORIES
	# -----------------
	wordsPerCat = [2] * 3 + [3] * 3
	wordsPerTrial = np.sum(wordsPerCat)

	# SET N-s
	# -------
	Ns = np.fromfile('N.txt', sep = '\t', dtype = int)
	Ns = np.concatenate(([2,3,4], Ns))
	print len(Ns)
	print len(df)


	for t in range(len(df)):
	    
	    # SETUP
	    # -----
	    chosenWords = [0] * exp['catNum']
	    chosenLastWord = [-1] * exp['catNum']
	    trialWords  = []
	    N = Ns[t]
	    
	    # randomize words per category
	    np.random.shuffle(wordsPerCat)

	    # chose N categories
	    ChosenCat = nSmallest(presented['asTargetCategory'], N)
	    
	    # for each category choose words 
	    for c in range(exp['catNum']):

	        # 1. get words
	        activeWords = exp['wordsCat'] == c
	        takeWords   = wordsPerCat[c]

	        if c in ChosenCat: 
	            chosenWords[c] =  nSmallest(presented['asTarget'], takeWords, activeWords)

	            # select last word
	            lastWords = np.zeros(np.shape(exp['wordsCat']), dtype = bool)
	            lastWords[chosenWords[c]] = True
	            chosenLastWord[c] = nSmallest(presented['asLast'], 1, lastWords)[0]
	        else:
	            chosenWords[c] =  nSmallest(presented['asNonTarget'], takeWords, activeWords)

	        # add chosen words to trial words
	        trialWords += chosenWords[c]

	    # ensure trialWords are numpy array
	    trialWords = np.array(trialWords)
	    
	    # shuffle trialWords
	    np.random.shuffle(trialWords)
	    wordIsTarget = np.zeros(len(trialWords), dtype = bool)
	    
	    # ensure chosenLastWord is last
	    # -----------------------------
	    for c in ChosenCat:
	        whereWords, = np.nonzero(np.in1d(trialWords, chosenWords[c]))
	        wordIsTarget[whereWords] = True
	        catOrder   = trialWords[whereWords]

	        if not catOrder[-1] == chosenLastWord[c]:
	            
	            whereIsLast, = np.nonzero(catOrder == chosenLastWord[c])[0]

	            # correct this
	            trialWords[whereWords[-1]] = chosenLastWord[c]
	            trialWords[whereWords[whereIsLast]] = catOrder[-1]

	    
	    # fill the DataFrame
	    # ------------------
	    
	    # N chosen cats & filename
	    df.iloc[t]['N'] = N
	    df.iloc[t]['ifExp'] = True if t + 1 > numTrain else False
	    df.iloc[t]['soundFile'] = subj['symb'] + subj['indTxt'] + '_' + \
	    						  fillz(t + 1 - numTrain, 2) + '.wav'
	    
	    # categories
	    df.iloc[t][colNamesCat[0:N]] = exp['categ'][ChosenCat]
	    
	    # last words
	    df.iloc[t][colNamesLast[0:N]] = data[u'słowa'].iloc[np.array(chosenLastWord)[ChosenCat]]
	    
	    # words (OR word numbers)
	    df.iloc[t][colNamesWord] = data[u'słowa'][trialWords]
	    
	    # is target
	    df.iloc[t][colNamesTarg] = wordIsTarget

	return df

# get user name:
def GetUserName():
    '''
    PsychoPy's simple GUI for entering 'stuff'
    Does not look nice or is not particularily user-friendly
    but is simple to call.
    Here it just asks for subject's name/code
    and returns it as a string
    '''
    myDlg = gui.Dlg(title="Pseudonim", size = (800,600))
    myDlg.addText('Podaj numer osoby badanej')
    myDlg.addField('numer')
    myDlg.show()  # show dialog and wait for OK or Cancel

    if myDlg.OK:  # the user pressed OK
        dialogInfo = myDlg.data
        try:
        	user = int(dialogInfo[0])
        except (ValueError, TypeError):
        	user = None
    else:
        user = None
   
    return user

# --------------------------


# =========
# main code
# =========


# SETUP
# -----
pth = os.getcwd()
clock = core.Clock()
t = clock.getTime()

# get subject info
# ----------------
subj = {}
subj['symb']   = 'KT'
subj['ind'] = GetUserName()
if subj['ind'] == None:
	# get free name
	subj['ind'], subj['subfl'] = free_filename(
		os.path.join(pth, 'output'), subj)
else:
	# check if not 
	isok = free_filename(os.path.join(pth, 'output'), subj, givenew = False)
	if not isok:
		subj['ind'], subj['subfl'] = free_filename(
			os.path.join(pth, 'output'), subj)

subj['indTxt'] = fillz(subj['ind'], 2)
subj['file'] = subj['symb'] + '_' + subj['indTxt']

# create subject directory:
os.mkdir(os.path.join(pth, 'output', subj['file']))

# CHECK if subject did finish - if not continue procedure

# get xls data
# ----------------

# check for files ending with .xls or .xlsx
# get data from xls
# CHANGE - now only one (first) file is taken (could stay this way)
fls       = os.listdir(pth)
xls_fl    = [f for f in fls if f.split('.')[-1] in ['xls', 'xlsx']]
full_path = os.path.join(pth, xls_fl[0])
data      = pd.io.excel.read_excel(full_path, u'słowa')


# organize into categories
exp['categ']      = list(set(data['kategorie']))
exp['catNum']     = len(exp['categ'])
exp['wordsNum']   = len(data)
# wordsInCat = wordsNum / catNum

# create numpy array mapping from categories to words:
exp['wordsCat'] = np.zeros(exp['wordsNum'], dtype = int)

for i in range(exp['catNum']):
    exp['wordsCat'][ data[data['kategorie'] == exp['categ'][i]].index ] = i



# generate trials:
trialInfo = generateTrials(subj, exp)


# SETUP DISPLAY
# -------------

# create a window
# CHANGE window settings
exp['window'] = visual.Window([800,600],monitor="testMonitor", 
    units="deg", color = [-0.2, -0.2, -0.2], fullscr=True)
exp['window'].mouseVisible = False

# ==
# get frame rate
print "testing frame rate..."
frame_rate = exp['window'].getActualFrameRate(nIdentical=25)
frame_time = 1000.0 / frame_rate
print "frame rate: " + str(frame_rate)
print "time per frame: " + str(frame_time)

# ==
# set time in frames according to frame rate
exp['frames'] = ms2frames(exp['times'], frame_time)


# ==
# create stimuli:
stim = {}
# CHANGE units from 'norm' to 'angle'?
stim['centerText'] = visual.TextStim(exp['window'], text='', font='Courier New', 
                                     pos=(0.0, 0.0), units = 'norm', height = 0.1)
stim['centerPolecenie'] = visual.TextStim(exp['window'], text='?', font='Courier New', 
                                     pos=(0.0, 0.0), units = 'norm', height = 0.1)

# below we create text for categories so that
# stim['categoryText'][N][i] gives i-th category 
# text when there are N categories
stim['categoryText'] = [[], [], [], [], []]
for N in [2, 3, 4]:
	unitDist = 2.0 / (N + 1)
	
	for i in range(N):
		pos = (unitDist * (i + 1) - 1.0, exp['categoriesHPos'])
		stim['categoryText'][N].append( visual.TextStim(exp['window'], 
			                            text='', font='Courier New', 
                                        pos=pos, 
                                        units = 'norm', height = 0.1) )


# BEGIN DISPLAY
# -------------

# first, save the data
trialInfo.to_excel(os.path.join(pth, 'output', subj['file'] + '.xls'))

# recreate column names
colNamesCat  = ['targetCategory' + fillz(x, 2) for x in range(1,5)]
colNamesLast = ['lastWordCat' + fillz(x, 2) for x in range(1,5)]
colNamesWord = ['word' + fillz(x, 2) for x in range(1,16)]
colNamesTarg = ['wordIsTarget' + fillz(x, 2) for x in range(1,16)]

# colNames = ['ifExp', 'N'] + colNamesCat + colNamesLast + \
# 	colNamesWord + colNamesTarg + ['soundFile']


def draw_frames(stimList, stimText, n_frames, win):
	# set text:
	for stim, text in zip(stimList, stimText):
		stim.setText(text)

	# display
	for frame in range(n_frames):
		for stim in stimList:
			stim.draw()

		# flip window
		win.flip()

for trial in range(len(trialInfo)):
	# get category text
	N          = trialInfo.iloc[trial]['N']
	categories = trialInfo.iloc[trial][colNamesCat[0:N]]
	words      = trialInfo.iloc[trial][colNamesWord]

	# shuffle categories but ensure they remain a list
	np.random.shuffle(categories)
	categories = list(categories)

	# DISPLAY TRIAL
	# -------------

	# 1. show categories
	stimList = stim['categoryText'][N]
	stimText = categories
	n_frames = exp['frames']['categoriesFirst']

	draw_frames(stimList, stimText, n_frames, exp['window'])


	# then, go through words and show word / fix
	for w in words:

		# 1. show fixation and categories
		stimList = [stim['centerText']] + stim['categoryText'][N]
		stimText = ['+'] + categories
		n_frames = exp['frames']['fixation']

		draw_frames(stimList, stimText, n_frames, exp['window'])

		# 2. show word and categories
		stimList = stim['categoryText'][N] + [stim['centerText']]
		stimText = categories + [w]
		n_frames = exp['frames']['word']

		draw_frames(stimList, stimText, n_frames, exp['window'])

	# prepare sound recording
	stim['centerPolecenie'].draw()
	exp['window'].flip()
	soundfile = os.path.join(pth, 'output', subj['file'], 
		trialInfo.iloc[trial]['soundFile'])

	# reset clock
	clock.reset()
	# record sound
	tm = record_to_file(soundfile, clock)

	# add timing info to trialInfo
	trialInfo.iloc[trial]['micInitTime'] = tm[0]
	trialInfo.iloc[trial]['RT'] = tm[1]


	# display feedback if training trial:
	if not trialInfo.iloc[trial]['ifExp']:
		# show feedback
		stimList = stim['categoryText'][N] + [stim['centerText']]
		stimText = list(trialInfo.iloc[trial][colNamesLast[0:N]]) + [u'poprawna odpowiedź to:']
		n_frames = exp['frames']['feedback']

		draw_frames(stimList, stimText, n_frames, exp['window'])

	# save the datafame every trial:
	trialInfo.to_excel(os.path.join(pth, 'output', subj['file'] + '.xls'))

	keys = event.getKeys()
	if 'q' in keys:
		core.quit()

core.quit()