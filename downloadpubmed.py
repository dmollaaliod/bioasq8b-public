"Download all PUBMED files referred in the corpus snippets"
import json
import urllib.request
import os

def alluri(corpusFile):
    """Return a set with the URI of all snippets

    >>> uriset = sorted(alluri('BioASQ-trainingDataset4b.json'))
    >>> len(uriset)
    12232
    >>> list(uriset)[:3]
    ['http://www.ncbi.nlm.nih.gov/pubmed/10024311', 'http://www.ncbi.nlm.nih.gov/pubmed/10024707', 'http://www.ncbi.nlm.nih.gov/pubmed/10024886']
"""
    with open(corpusFile) as f:
        data = json.load(f)
    result = set()
    for q in data['questions']:
        if 'snippets' in q:
            result = result | set(s['document'] for s in q['snippets'])
    return result

    return set(s['document']
               for q in data['questions']
               for s in q['snippets']
               if 'snippets' in q)

def fetchpubmed(uri):
    """Return the XML contents of the PubMed document

    >>> s = fetchpubmed('17671148')
    >>> len(s)
    8736
    >>> str(s).find('<PMID Version="1">17671148</PMID>') > 0
    True
"""
    f = urllib.request.urlopen("http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=%s&retmode=xml" % uri)
    return f.read()

if __name__ == "__main__":
    import doctest
    doctest.testmod()

    import os.path
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", '--path',
                        default="Task5bPubMed",
                        help="Path where the PubMed articles will be saved")
    parser.add_argument("-i", '--inputfile',
                        default="BioASQ-trainingDataset5b.json",
                        help="Input JSON file")
    args = parser.parse_args()

    sofar = 0
    alluris = alluri(args.inputfile)
    for d in alluris:
        sofar +=1
        filename = os.path.join(args.path, os.path.basename(d))+'.xml'
        print(filename)
        if os.path.exists(filename):
            continue
        print(" %i %%: %s" % (sofar*100/len(alluris), d))
        xmltext = fetchpubmed(os.path.basename(d))
        with open(filename,'wb') as f:
            f.write(xmltext)
