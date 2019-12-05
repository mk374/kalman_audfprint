
#We can begin writing code that reflects the functionality of the Kalman filter
#without the mathematical baggage. We can simply begin to write simple
#heuristics that imitate the filter.

# import audfprint
import pyaudio
import wave
import sys
import audioop
import struct
import numpy as np
import time


import audio_read
import audfprint_analyze
import hash_table
import audfprint_match


def kalman_pseudo(data_input, prev_data):
    """
    data_input represents the ranking of hashes
    (modify input to represent sorted list of tuples)
    i.e. data_input = [(file_name, percentage of common hashes),()]

    prev_data will represent the previous recording's hashes. 
    i.e. prev_data = file_name
    """
    if prev_data in [x for x,y in data_input]:
        return prev_data
    return data_input[0][0]



def main():
    #call in the python audfprint.py file and then get the required stuff.

    pass


def different_record(SECONDS):
    CHUNK = 1024 
    FORMAT = pyaudio.paFloat32 #paInt8
    CHANNELS = 1
    RATE = 11025 #sample rate
    RECORD_SECONDS = SECONDS
    WAVE_OUTPUT_FILENAME = "output.wav"

    p = pyaudio.PyAudio()


    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    # FRAMES SAVES A WAVE FILE. IF YOU WANT TO CHECK THE CODE WITH JS
    #UNCOMMENT THE UNDERLINED

    #_________________________
    frames = []

    ampCheck = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
        ampCheck.extend(audio_read.buf_to_float(data)[1::2]) # 2 bytes(16 bits) per channel

    print("* done recording")

    
    # print(ampCheck[-10:])
    # print(len(ampCheck))

    stream.stop_stream()
    stream.close()
    p.terminate()

    # _______________________________________________
    # wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    # wf.setnchannels(CHANNELS)
    # wf.setsampwidth(p.get_sample_size(FORMAT))
    # wf.setframerate(RATE)
    # wf.writeframes(b''.join(frames))
    # wf.close()

    return ampCheck

def match_file(matcher, analyzer, ht, filename, hashList):
    """ Read in an audio file, calculate its landmarks, query against
        hash table.  Return top N matches as (id, filterdmatchcount,
        timeoffs, rawmatchcount), also length of input file in sec,
        and count of raw query hashes extracted
    """
    q_hashes = hashList
    # Fake durations as largest hash time
    if len(q_hashes) == 0:
        durd = 0.0
    else:
        durd = analyzer.n_hop * q_hashes[-1][0] / analyzer.target_sr
    
    # Run query
    rslts = matcher.match_hashes(ht, q_hashes)
    # Post filtering
    if matcher.sort_by_time:
        rslts = rslts[(-rslts[:, 2]).argsort(), :]
    return rslts[:matcher.max_returns, :], durd, len(q_hashes)

def file_match_to_msgs(matcher, analyzer, ht, filename, hashList):
    """ Perform a match on a single input file, return list
        of message strings """
    rslts, dur, nhash = match_file(matcher, analyzer, ht, filename, hashList)
    t_hop = analyzer.n_hop / analyzer.target_sr
    
    qrymsg = filename
    msgrslt = []
    if len(rslts) == 0:
        # No matches returned at all
        nhashaligned = 0
        print("no matches")
    else:
        for (tophitid, nhashaligned, aligntime, nhashraw, rank,
                min_time, max_time) in rslts:
            # figure the number of raw and aligned matches for top hit
            if True:
                if matcher.find_time_range:
                    msg = ("Matched {:6.1f} s starting at {:6.1f} s in {:s}"
                            " to time {:6.1f} s in {:s}").format(
                            (max_time - min_time) * t_hop, min_time * t_hop, filename,
                            (min_time + aligntime) * t_hop, ht.names[tophitid])
                else:
                    msg = "Matched {:s} as {:s} at {:6.1f} s".format(
                            qrymsg, ht.names[tophitid], aligntime * t_hop)
                msg += (" with {:5d} of {:5d} common hashes"
                        " at rank {:2d}").format(
                        nhashaligned, nhashraw, rank)
                msgrslt.append(msg)
            else:
                msgrslt.append(qrymsg + "\t" + ht.names[tophitid])
            
    return msgrslt, rslts

    
def kalman_matching():
    #get find_peaks from analyze
    analyzer = audfprint_analyze.Analyzer()
    hash_tab = hash_table.HashTable('fpdbase.pklz')
    matcher = audfprint_match.Matcher()

    sampling_seconds = 10
    sampling_interval = 15
    prev_resultID = []

    count = 0

    true_start = time.time()

    print("click now!")
    time.sleep(1)

    while True:
        start = time.time()
        twoSecondArray = different_record(sampling_seconds)

        peakLists = analyzer.find_peaks(twoSecondArray, 11025)
        landmarkLists = analyzer.peaks2landmarks(peakLists)
        hashesLists = audfprint_analyze.landmarks2hashes(landmarkLists)

        hashes_hashes = (((hashesLists[:, 0].astype(np.uint64)) << 32)
                            + hashesLists[:, 1].astype(np.uint64))
        unique_hash_hash = np.sort(np.unique(hashes_hashes))
        unique_hashes = np.hstack([
            (unique_hash_hash >> 32)[:, np.newaxis],
            (unique_hash_hash & ((1 << 32) - 1))[:, np.newaxis]
        ]).astype(np.int32)
        hashes = unique_hashes
        #now the matching
        # for num, filename in enumerate(filename_iter):
        #     # count += 1
        #     msgs = matcher.file_match_to_msgs(analyzer, hash_tab, filename, num)
        #     report(msgs)

        # file_match_to_msgs(self, analyzer, ht, qry, number=None)
        # print(matcher.file_match_to_msgs(analyzer, hash_tab, "Some qry name"))
        # rslts, dur, nhash = match_file(matcher, analyzer, hash_tab, "some query", hashesLists)
        message, results = file_match_to_msgs(matcher, analyzer, hash_tab, "FROM MICROPHONE", hashes)
        print(message)
        if len(results) == 0:
            sampling_interval -= 5
            if sampling_interval <= 10:
                sampling_interval = 10
            sampling_seconds += 5
        elif results[0][0] == prev_resultID:
            #increase sampling_interval
            sampling_interval +=5
            sampling_seconds -= 5
            if sampling_seconds <= 10:
                sampling_seconds = 5
            prev_resultID = results[0][0]
        else:
            sampling_interval -= 5
            if sampling_interval <= 10:
                sampling_interval = 10
            sampling_seconds += 5

            prev_resultID = results[0][0]
# (tophitid, nhashaligned, aligntime, nhashraw, rank,
#                 min_time, max_time)
        print(sampling_seconds, sampling_interval)
        
        count += 1
        end = time.time() - start
        print("how much this iteration took", end)
        print("how many seconds so far", time.time() - true_start)

        time.sleep(sampling_interval - (end - sampling_seconds))
    print(count)

def regular_matching():
    #get find_peaks from analyze
    analyzer = audfprint_analyze.Analyzer()
    hash_tab = hash_table.HashTable('fpdbase.pklz')
    matcher = audfprint_match.Matcher()

    sampling_seconds = 10
    sampling_interval = 15
    prev_resultID = []

    count = 0
    while True:
        start = time.time()
        twoSecondArray = different_record(sampling_seconds)

        peakLists = analyzer.find_peaks(twoSecondArray, 11025)
        landmarkLists = analyzer.peaks2landmarks(peakLists)
        hashesLists = audfprint_analyze.landmarks2hashes(landmarkLists)
        print(hashesLists)

        hashes_hashes = (((hashesLists[:, 0].astype(np.uint64)) << 32)
                            + hashesLists[:, 1].astype(np.uint64))
        unique_hash_hash = np.sort(np.unique(hashes_hashes))
        unique_hashes = np.hstack([
            (unique_hash_hash >> 32)[:, np.newaxis],
            (unique_hash_hash & ((1 << 32) - 1))[:, np.newaxis]
        ]).astype(np.int32)
        hashes = unique_hashes
        #now the matching
        # for num, filename in enumerate(filename_iter):
        #     # count += 1
        #     msgs = matcher.file_match_to_msgs(analyzer, hash_tab, filename, num)
        #     report(msgs)

        # file_match_to_msgs(self, analyzer, ht, qry, number=None)
        # print(matcher.file_match_to_msgs(analyzer, hash_tab, "Some qry name"))
        # rslts, dur, nhash = match_file(matcher, analyzer, hash_tab, "some query", hashesLists)
        message, results = file_match_to_msgs(matcher, analyzer, hash_tab, "FROM MICROPHONE", hashes)
        print(sampling_seconds, sampling_interval)
        
        count += 1
        end = time.time() - start
        print(end)
        
        time.sleep(sampling_interval - (end - sampling_seconds))
        
    print(count)
if __name__ == "__main__":
    kalman_matching()
