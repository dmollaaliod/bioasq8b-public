"""A basic summariser that returns the n best candidates"""

print("Basic summariser")

def answersummaries(questions_and_candidates, extractfeatures, predict):
    """Batch process to generate several answer summaries"""
    questions_list = []
    candidates_sentences_list = []
    candidates_sentences_ids = []
    for q_c_n in questions_and_candidates:
        if q_c_n == None:
            continue
        question, candidates_sentences, n = q_c_n
        questions_list += [question] * len(candidates_sentences)
        # print("Question: %s" % question)
        # print("N=%i" % n)
        # print("There are %i candidates sentences" % len(candidates_sentences))
        # print("Candidates sentences[0:2]:")
        # print(candidates_sentences[0:2])
        candidates_sentences_list += [c[0] for c in candidates_sentences]
        candidates_sentences_ids += [[c[1]] for c in candidates_sentences]

    features = extractfeatures(questions_list, candidates_sentences_list)
    predictions = [p[0] for p in predict(features[0], features[1], candidates_sentences_ids)]
    result = []
    this_i = 0
    for q_c_n in questions_and_candidates:
        if q_c_n == None:
            # result.append(None)
            continue
        # print(q_c_n)
        question, candidates_sentences, n = q_c_n
        next_i = this_i + len(candidates_sentences)
        scores = list(zip(predictions[this_i:next_i],
                          range(len(candidates_sentences))))
        scores.sort()
        summary = scores[-n:]
        summary.sort(key=lambda x: x[1])
        # print("this_i=%i, next_i=%i, len(candidates_sentences)=%i" % (this_i, next_i, len(candidates_sentences)))
        # print("Summary:")
        # print(summary)
        result.append([candidates_sentences[i][0] for score, i in summary])
        this_i = next_i
    return result
