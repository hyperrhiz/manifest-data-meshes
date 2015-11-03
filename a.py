

"""\begin{figure*} 
~\hrule~
\begin{minipage}[t]{.45\linewidth}
\scriptsize\vspace{1mm}
{\bf The Line class (as described in \tion{mm}):}

 
\begin{python}[left]"""
class Line:
  """The central data structure of HOW. Training data 
  is converted into a set of Lines, each of which is 
  defined by a 'X,Y' pair, where 'X'
  has better scores than 'Y'."""
  all   = []    # where to store all the lines
  weights=[]    # (weight,feature) pairs, sorted
  using = []    # what columns to use
  F     = 0.5   # magnitude control parameter
  B     = 50    # ratio of how many columns to displace

  def __init__(i,x,y):
    if x.score < y.score:
      x,y = y,x      # x must have best score
    i.x, i.y, i.c = x, y, dist(x,y) 
    Line.all += [i]  # remember this line

  def displace(i,z):
    for j in Line.using:
      z[j] = z[j] + Line.F*(x[j] - z[j])
    return z

  def dist(i,z):
    _,out = geometry(i.x,i.y,z)
    return out
"""\end{python}

~\colorrule{gray}~

{\bf Main loop, of the  HOW  instance-based planner:}

\begin{python}[left]"""
def HOW(training, testing, b = Line.B):
  """ Testing data is displaced towards
  the X point of the nearest Line."""
  def train(data):
    Line.weights= weightedFeatures(data)
    b           = int(len(Line.weights)*B/100)
    Lines.using = Line.weights[:b]
    clusters    = [k for k in 
                   leafs(cluster(data,len(data)))]
    for cluster in clusters:
      one = exemplar(cluster)
      two = exemplar(nearest(one, clusters))
      Line(one,two)
 
  def exemplar(cluster):
    "Apply the rules at end of section II.B"
 
  def nearestLine(z):
    out,least=None,10000
    for line in Line.all:
      d = line.dist(z)
      if d < least: out,least = line,d
    return out
  # ---- begin main code for  'HOW' ---------------
  train(training)
  return [nearestLine(z).displace(z) for z in testing]
"""\end{python}

~\colorrule{gray}~

{\bf Misc utilities; e.g. code for geometry (as described in \fig{where}c):}

 
\begin{python}[left]"""
def geometry(x, y, c, z): 
  a = dist(z,x)
  b = dist(z,y) 
  x = (a**2 + c**2 - b**2)/(2*c) 
  y = (a**2 - max(x,0))**0.5
  return x,y

def dist(x,y):
   tmp = sum(w*(x[j] - y[j])**2 for w,j in Line.using) 
   return tmp**0.5 / len(Line.using)**2
   
def nearest(x,data):
  return furthest(x,data,best=1000,
                    better=lambda j,k:j < k)
 
def furthest(x,data, best= -1, 
                     better=lambda j,k:j > k):  
  out = None
  for one in data:
    d = dist(one,x)
    if gt(d,best): out, best = y, d
  return out

def leafs(data):
  leaf = True
  for more in data.leafs:
    if more:
       leaf = False
       for one in leafs(more):
         yield one
  if leaf: 
    yield data
"""\end{python}
 \end{minipage}~~~~~~~~~\begin{minipage}[t]{.45\linewidth} 
\scriptsize\vspace{1mm}
{\bf Clustering (as described in \fig{where}a):}
\begin{python}[right]"""
def cluster(data, n,lvl=100):

  def splitAcross2Points(data): 
    tmp = random.choose(data)
    x = furthest(tmp, data)
    y = furthest(x, data) 
    c = dist(x,y)  
    if x.scores < y.scores:
      x,y = y,x 
    for one in data.members: 
      one.pos = geometry(x,y,c,one)
    data = sorted(data) # sorted by 'pos.x'
    return x, y, split(data)
  
  def split(data):   
    mid = len(data)/2; 
    return data[mid:], data[:mid]
    
  # --- begin main code for  'cluster' --------
  if lvl < 1: 
     return data # stop if out of levels
  leafs = [] # Empty Set
  x,y,left,right = splitAcross2Points(data) 
  if len(left) > sqrt(n):  
     leafs += cluster(left, n, lvl - 1)  
  if len(right) > sqrt(n):
     leafs += cluster(right,n,  lvl - 1) 
  date.leafs = leafs
  return leafs
"""\end{python} 

~\colorrule{gray}~

{\bf Feature weighting  (as described in \fig{where}d):}

\begin{python}[right]"""
def weightedFeatures(cols):
  "Returns col indexes, sorted by their weight."
  
  class Stats():  
    """Utility class. Handles incremental update of
       n, mean, standard dev."""
    def __init__(i,inits=[]):
      i.n = i.mu = i.m2 = 0.0
      map(i.__add__,inits)
      
    def sd(i) :  
      return (max(0.0,i.m2)/(i.n - 1))**0.5
      
    def __add__(i,x):
      i.n  += 1
      delta = x - i.mu
      i.mu += delta/(1.0*i.n)
      i.m2 += delta*(x - i.mu) 
      
    def __sub__(i,x):
      i.n  -= 1
      delta = x - i.mu
      i.mu -= delta/(1.0*i.n)
      i.m2 -= delta*(x - i.mu) 
      
  def divide(this,tiny=2):
    "Find the split that most reduces std dev."
    lhs, rhs = Stats(), Stats(x[1] for x in this)
    n, least, cut = rhs.n*1.0, rhs.sd(), None
    for j,x in enumerate(this):
      if lhs.n > tiny and rhs.n > tiny:
        tmp = lhs.n/n*lhs.sd() + rhs.n/n * rhs.sd()
        if tmp < least:
           cut,least = j,tmp
      rhs - x[1]
      lhs + x[1]
    return cut,least
    
  def recurse(this,cuts):
    cut,sd = divide(this)
    if cut:
      recurse(this[:cut],cuts)
      recurse(this[cut:],cuts)
    else:
      cuts += [(sd,len(this))]
    return cuts  
    
  def weight1(i,obj,col):
    "Returns weight of column 'i'."
    pairs = sorted((row[i],row[obj]) for row in col)
    n     = length(col)
    w     = sum(v*n1/n for v,n1 in recurse(pairs,[]))
    return w,i
    
  # ---- begin main code for 'featureWeighting' -----
  obj = len(cols[0]) - 1 
  return sorted(weight1(col,i,obj)
                for i,col in enumerate(cols) 
                if not i == obj)

"""\end{python}
\end{minipage}
~\hrule~
\caption{HOW (Python-style psuedo-code).
For brevity's sake, this code skips certain low-level details.
For a full working implementation, see https://github.com/ai-se/HOW1/src.
Functions shown in \textcolor{blue}{{\bf blue}} are defined somewhere in this figure.}\label{fig:howcode}   
\end{figure*}
"""
