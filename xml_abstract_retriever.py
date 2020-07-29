
from xml.dom.minidom import parse, parseString
import os
import codecs

def getText(nodelist):
    """Return the concatenated plain text of all nodes
    Taken from http://docs.python.org/library/xml.dom.minidom.html"""
    rc = ""
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            rc = rc + node.data
    return rc

def getAbstract(filename,version="1.0",verbose=1):
    """Get the abstract if the associated XML file exists and has an abstract
    Otherwise return an empty string

    The abstract is a pair (text,sections) with this information:
     - text: A concatenation of the strings
     - sections: a list of dictionaries with these keys:
          'offset': a touple of begin-end string offsets
          'label': section label
          'nlmcategory': NLM category, which is one of:
             BACKGROUND, CONCLUSIONS, METHODS, OBJECTIVE, RESULTS
        If the list is empty then there is no sectioning information.
    >>> a1 = getAbstract('Task4bPubMed/30666.xml', version="0")
    >>> a1[0][:40]
    'Hemolytic anemia is a well-recognized co'
    >>> a1[1]
    [{'offset': (0, 833)}]
    >>> a2 = getAbstract('Task4bPubMed/10024707.xml', version="0")
    >>> a2[0][:40]
    'Tobacco smoking is a major cause of prev'
    >>> a2[1][:2] == [{'label': 'BACKGROUND', 'nlmcategory': 'BACKGROUND', 'offset': (0, 235)}, {'label': 'AIM', 'nlmcategory': 'OBJECTIVE', 'offset': (235, 313)}]
    True
    """
    #print("Getting abstract, version",version)
    if os.path.exists(filename):
        # Read from iso-8859-1 encoding and convert to utf-8 for XML processing
        string = codecs.open(filename,'r','iso-8859-1').read()
        string = string.encode('utf-8')
        try:
            d = parseString(string)
        except:
            if verbose>0:
                print('Exception: XML file malformed')
            return ('',{})
        if version == "0":
            abstractkey = 'AbstractText'
            labelkey = 'Label'
            categorykey = 'NlmCategory'
        else:
            abstractkey = 'abstracttext'
            labelkey = 'label'
            categorykey = 'nlmcategory'
        abstract = d.getElementsByTagName(abstractkey)
        if len(abstract) > 0:
            sections = []
            text = ""
            for a in abstract:
                att = a.attributes
                begin = len(text)
                text += " "+getText(a.childNodes).strip()
                end = len(text)
                if att.getNamedItem(labelkey) and att.getNamedItem(categorykey):
                    label = att.getNamedItem(labelkey).nodeValue
                    nlmcategory = att.getNamedItem(categorykey).nodeValue
                    sections.append({'offset':(begin,end),
                                     'label':label,
                                     'nlmcategory':nlmcategory})
                else:
                    sections.append({'offset':(begin,end)})

            #print('Abstract OK')
            return (text.strip(),sections)
        else:
            #print('Empty abstract')
            return ('',{})
    else:
        if verbose>0:
            print('XML file not found:', filename)
        return ('',{})

if __name__ == "__main__":
    import doctest
    doctest.testmod()
