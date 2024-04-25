Knowledge Base Strings
=======================
String manipulation (operators, functions)
---------------------------------------------
| s[2], len(s), s+t, s[2:4], s in t, s.strip(), s.lower(), s.upper() 
| s.startswith(prefix), s.endswith(suffix), 
| 'some, and'.split(','),
| 3 * '01',

>>> ','.join(('word1', 'word2'))
'word1,word2'
>>> 'apples,pies'.split(',')
['apples', 'pies']
>>> 'first: {fname}, last: {lname}'.format(fname='Julie', lname='Frankopone')
'first: Julie, last: Frankopone'
# Test if value is alphanumeric.
>>> a = 'a2'
>>> a.isalnum()
True

| ``isalpha()``   Returns True if the string consists only of letters and isn’t blank
| ``isalnum()``   Returns True if the string consists only of letters and numbers and is not blank

Strings are immutable
-------------------------
Operations like s = s[1:] or s +=' 123' imply creating a new array of characters 
that is then assigned back to s. Or alternatively use list().

String immutability also means O(n**2) time to concatenate a character to a 
string n times in a for loop (unless when some Python built-ins improve it to O(n)).

Keep in mind that updating from the front is more expensive that writing to the end
(and reversing the result when needed as the last operation).


















