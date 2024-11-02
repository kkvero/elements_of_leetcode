
Stack and Queue Questions Part 1
================================
173. Print elements of a linked list
---------------------------------------
| Using a stack to print the entries of a singly linked list.
| T O(N), S O(N). 
| (We could also use the technique of reversing a linked list, then T O(N), S O(1).)

::

    def print_linked_list_in_reverse(head):
        nodes = []
        while head:
            nodes.append(head.data)
            head = head.next
        while nodes:
            print(nodes.pop())

174. (LC 155) Min Stack
----------------------------
`155. Min Stack <https://leetcode.com/problems/min-stack/description/>`_
Medium

| We can improve on space: 
| -if a new element pushed to stack is greater then current max, then we don't have to
| add it to minStack (it will never be min).
| -record min and min_count. So if elem=5 and we encounter a second 5, we record count+=1
| for elem 5. E.g.:
| stack = [2,2,1,4]
| minStack = [(2,1), (2,2), (4,1)] 
| #can be tuples, named tuples, or class within our Stack class.

**Solution 1** [:ref:`10 <ref-label>`] [:ref:`7 <ref-label>`] T O(1), S O(N) ::

    class MinStack:
        def __init__(self):
            self.stack = []
            self.minStack = []

        def push(self, val: int) -> None:
            self.stack.append(val)
            val = min(val, self.minStack[-1] if self.minStack else val)
            self.minStack.append(val)

        def pop(self) -> None:
            self.stack.pop()
            self.minStack.pop()

        def top(self) -> int:
            return self.stack[-1]

        def getMin(self) -> int:
            return self.minStack[-1]

175. (LC 150) Evaluate Reverse Polish Notation
--------------------------------------------------
`LC 150. Evaluate Reverse Polish Notation <https://leetcode.com/problems/evaluate-reverse-polish-notation/description/>`_
Medium

| RPN (Reverse Polish Notation):
| Example 1:
| Input: ``tokens = ["2","1","+","3","*"]``
| Output: 9
| Explanation: ((2 + 1) * 3) = 9
 
| Ex 2:
| Input: ``tokens = ["4","13","5","/","+"]``
| Output: 6
| Explanation: (4 + (13 / 5)) = 6

| **Recognize the simplicity of the RPN mechanism:**
| No matter what form of RPN you are given, even if [num, num, num, oper, oper],
| the procedure is the same:
| -the partial results are added and removed in LIFO order, so use stack
| -if cur char == num: then add to stack
| if cur char == operator: then pop the last two nums from stack and use that operator on them,
| add result to stack.
 
| Approaches:
| The differences in implementation concern only the way you implement the operators:
| -don't implement, use directly +,-,*,/
| -use hash, map using lambda
| -use hash, map to operator module functions
| -the last entry in stack is the result of the last expression = our answer

.. admonition:: Recall: passing args to lambda

    a = lambda x, y: x + y
    print(a(2, 3))

**Solution 1** [:ref:`2 <ref-label>`] (LC accepted 95,85%) ::

    stack = []  # intermidiate results
    DELIMITER = ","
    OPERATORS = {
        "+": lambda x, y: x + y,
        "-": lambda x, y: y - x,
        "*": lambda y, x: x * y,
        "/": lambda x, y: int(y / x),
    }

    for token in tokens.split(DELIMITER):
        if token in OPERATORS:
            stack.append(OPERATORS[token](stack.pop(), stack.pop()))
        else:  # token is a num
            stack.append(int(token))
    return stack[-1]

**Solution 2** [:ref:`7 <ref-label>`] ::

    import operator
    class Solution:
        def evalRPN(self, tokens: List[str]) -> int:
            opt = {
                "+": operator.add,
                "-": operator.sub,
                "*": operator.mul,
                "/": operator.truediv,
            }
            s = []
            for token in tokens:
                if token in opt:
                    s.append(int(opt[token](s.pop(-2), s.pop(-1))))
                else:
                    s.append(int(token))
            return s[0]

**Solution 3** [:ref:`10 <ref-label>`] O(N) ::
    
    class Solution:
        def evalRPN(self, tokens: List[str]) -> int:
            stack = []
            for c in tokens:
                if c == "+":
                    stack.append(stack.pop() + stack.pop())
                elif c == "-":
                    a, b = stack.pop(), stack.pop()
                    stack.append(b - a)
                elif c == "*":
                    stack.append(stack.pop() * stack.pop())
                elif c == "/":
                    a, b = stack.pop(), stack.pop()
                    stack.append(int(float(b) / a))
                else:
                    stack.append(int(c))
            return stack[0]


**My V** (LC accepted 48,85%) ::

    class Solution:
        def evalRPN(self, tokens: List[str]) -> int:
            operators = {'+': operator.add, '-': operator.sub,
                        '*': operator.mul, '/': operator.truediv}
            stack = []
            for s in tokens:
                if s in operators:
                    val1 = stack.pop()
                    val2 = stack.pop()
                    val3 = operators[s](val2, val1)
                    stack.append(int(val3))
                else:
                    stack.append(int(s))
            return stack[0]

176. (LC 71) Simplify Path
------------------------------
`71. Simplify Path <https://leetcode.com/problems/simplify-path/description/>`_
Medium

| **Key:**
| -Understand the task, it is simple.
| When given a path, it is a series of cd commands in the shell.
| E.g. path = '/../abc//./def/'
| 0)We always start at the root.
| 1) cd ..  (we are at the root, so .. does nothing.)
| 2) cd abc  (we are at /abc)
| 3) ignore //, ignore .
| 4) cd def (we are at /abc/def)
| Note: we always ignore //, and . (cd to current)
 
| -Why stack
| Because of ..
| When we encounter .. we go back one dir, so pop from stack ONCE.

Our new path IS STACK itself (stack is not a helper data structure).

**Solution 1** ::

    class Solution:
        def simplifyPath(self, path: str) -> str:
            stack = []
            for i in path.split("/"):
                #  if i == "/" or i == '//', it becomes '' (empty string)

                if i == "..":
                    if stack:
                        stack.pop()
                elif i == "." or i == '':
                    # skip "." or an empty string
                    continue
                else:
                    stack.append(i)

            res = "/" + "/".join(stack)
            return res

| Note when splitting on '/', we get empty ``''``:
| #1

>>> path = '/../ed//tr/'
>>> L = path.split("/")
>>> L
['', '..', 'ed', '', 'tr', '']

| #2
| To stack we just add, e.g. ``stack = ['ed', 'tr']``

**My V** (LC accepted 85, 37%) ::

    class Solution:
        def simplifyPath(self, path: str) -> str:
            stack = []
            path = path.split('/')
            for p in path:
                if p == '.':
                    continue
                elif p == '..':
                    if stack:
                        stack.pop()
                elif p != '/' and p != '':
                    stack.append(p)
            new_path = '/'.join(stack)
            return '/' + new_path

177. (LC 22) Generate Parentheses
------------------------------------
`22. Generate Parentheses <https://leetcode.com/problems/generate-parentheses/description/>`_
Medium

| **Background on Backtracking** 
| [`View the original article <https://www.simplilearn.com/tutorials/data-structure-tutorial/backtracking-algorithm>`_]
| Backtracking is a technique for listing all possible solutions for a combinatorial algorithm problem.

Backtracking is an algorithmic technique whose goal is to use brute force to find 
all solutions to a problem. It entails gradually compiling a set of all possible solutions. 
Because a problem will have constraints, solutions that do not meet them will be removed.

A backtracking algorithm uses the depth-first search method. 
If a proposed solution satisfies the constraints, it will keep looking. 
If it does not, the branch is removed, and <the algorithm returns to the previous level>.

State-Space Tree
A tree that represents all of the possible states of the problem, from the root as an 
initial state to the leaf as a terminal state.

Intermediate checkpoints. 
If the checkpoints do not lead to a viable solution, the problem can return to the 
checkpoints and take another path to find a solution.

**Backtracking boilerplate:** ::

    void FIND_SOLUTIONS(parameters):
        if (valid solution):
            store the solution
            Return
        for (all choice):
            if (valid choice):
            APPLY (choice)
            FIND_SOLUTIONS (parameters)
            BACKTRACK (remove choice)    <===
        Return

| **Our alg in terms of backtracking**
| So after finding and appending a valid solution. 
| res = ['((()))']
| We start to backtrack -> stack.pop()
| Till
| stack = ['(', '(']
| And then start to collect another valid answer.

| **Keys to our alg in general:**
| How do you decide which ( or ) to add to response?
| You keep track of the counts for the number of closing and opening parentheses.
| Initially openCount=0, closeCount=0
| You deal with 2 cases:
| 1)add '(' => if openCount < n
| '(('
| Put an open one any time, just not more than <n> given times.
| 2)add ')' => if openCount > closedCount
| '(' -> '()', ')' -> No
| Put closing one only if there are more open parentheses than closed ones.
| I.e. there is a pending open one yet to close.
 
Use variable <cur> where you build a current valid string
 
| **Keys:**
| -two ifs, not else
| -add closed if num of open > num closed
| -add open if num of open < total num of pairs

**Solution 2** (rewrite from C, LC accepted 30, 15%, 2nd run 70, 80) ::

    class Solution:
        def generateParenthesis(self, n: int) -> List[str]:
            cur = ''
            res = []
            def backtrack(cur, o, c):
                if o == n == c:
                    res.append(cur)
                    return
                # add open par
                if o < n:
                    backtrack(cur + '(', o+1, c)
                # add closed
                if o > c:
                    backtrack(cur + ')', o, c+1)
            
            backtrack(cur, 0,0)
            return res


**Solution 1** ::

    class Solution:
        def generateParenthesis(self, n: int) -> List[str]:
            stack = []
            res = []

            def backtrack(openN, closedN):
                if openN == closedN == n:
                    res.append("".join(stack))
                    return

                if openN < n:
                    stack.append("(")
                    backtrack(openN + 1, closedN)
                    stack.pop()
                if closedN < openN:
                    stack.append(")")
                    backtrack(openN, closedN + 1)
                    stack.pop()

            backtrack(0, 0)
            return res

| Complexity Time O((2**n) * n)
| Because we have 2 possibilities, closed or open (, ).

178. (LC 739) Daily Temperatures
---------------------------------------
`739. Daily Temperatures <https://leetcode.com/problems/daily-temperatures/description/>`_
Medium

| **Keys:**
| -Stack, mono decreasing, storing pairs [temp, index]
 
| Brute force would take us O(n**n).
| Using monotonic stack: O(n) Time&Space.

**Solution 1** [:ref:`10 <ref-label>`] ::

    class Solution:
        def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
            res = [0] * len(temperatures)         #1
            stack = []  # pair: [temp, index]

            for i, t in enumerate(temperatures):
                while stack and t > stack[-1][0]:
                    poppedT, poppedInd = stack.pop()
                    res[stackInd] = i - stackInd
                stack.append((t, i))
            return res

| #1
| After iterating over all temps, if something is still in the stack, then there is
| not a warmer day for these temps. Then the defaults of our ans 0 will be used.
 
| If we encounter temperature <greater than temp on stack>:
| -we pop from stack and 
| -calculate response for the index on stack
| /When no stack, put current (temp, index) to stack.
 
| **Going through the loop:**
| Input: temperatures = [73,74,75,71,69,72,76,73]
| Output: [1,1,4,2,1,1,0,0]
 
| 1)(73,0)
| stack=[]
| res = [0,0,0,0,0,0,0]
 
| stack=[(73,0)]
| res = [0,0,0,0,0,0,0]
 
| 2)(74,1)
| stack=[(73,0)]
| res = [0,0,0,0,0,0,0]
 
| stack=[]
| poppedT, pippedInd = 73, 0
| calculate res = curInd - poppedInd = 1-0=1
| At ind 0 = 1
| res = [1,0,0,0,0,0,0]
 
stack=[(74,1)]
 
| 3)(75,2)
| stack=[(74,1)]
| res = [1,0,0,0,0,0,0]
 
| calculate res = curInd - poppedInd = 2-1=1
| res = [1,1,0,0,0,0,0]
 
stack=[(75,2)]
 
| 4)(71,3)
| stack=[(75,2)]
| Cur val 71 NOT > 75, then we just put it on stack.
| stack=[(75,2), (71, 3)]
 
| 5)(69,4)
| Cur val 69 NOT > 71, then we put it on stack.
| stack=[(75,2), (71, 3), (69, 4)]
 
| 6)(72, 5)
| stack=[(75,2), (71, 3), (69, 4)]
| calculate res = curInd - poppedInd = 5-4=1
| At ind 4 = 1
| stack=[(75,2), (71, 3)]
| res = [1,1,0,0,1,0,0]
 
| still 72 > 71
| calculate res = curInd - poppedInd = 5-3=2
| At ind 3 = 2
| res = [1,1,0,2,1,0,0]
| stack=[(75,2)]
 
| 72 NOT > 75, add 72 to stack
| stack=[(75,2), (72,5)]
 
| 7)(76,6)
| stack=[(75,2), (72,5)]
| calculate res = curInd - poppedInd = 6-5=1
| At ind 5 = 1
| stack=[(75,2)]
| res = [1,1,0,2,1,1,0]
 
| still 76 > 75
| calculate res = curInd - poppedInd = 6-2=4
| At ind 2 = 4
| stack=[]
| res = [1,1,4,2,1,1,0]
| stack=[(76,6)]
 
| 8)(73,7)
| stack=[(76,6)]
| 73 NOT > 76, so we just put it to stack
| stack=[(76,6), (73,7)]

179. (LC 853) Car Fleet
-------------------------
`853. Car Fleet <https://leetcode.com/problems/car-fleet/description/>`_
Medium

| **My V** (LC accepted 65, 12)
| In data:
| position = [10, 8, 0, 5, 3]
| speed = [2, 4, 1, 1, 3]
| target = 12
 
| Make array cars = [(position, time)] -> [(10,1), (8,1), (0,12), (5,7), (3,3)]
| Sort cars on position, cars = [(0,12), (3,3), (5,7), (8,1), (10,1)]
| Loop start with the car closest to target, (10,1).
| Pop: time. If current time > previous time, then this car won't catch up with
| the previous car (because current is farther from target than previous, and 
| this farther car will take more time to reach the target).

::

    class Solution:
        def carFleet(self, target: int, position: List[int], speed: List[int]) -> int:
            time = [0] * len(position)
            cars = [0] * len(position)
            for i in range(len(position)):
                time[i] = (target - position[i]) / speed[i]
                cars[i] = (position[i], time[i])
            cars.sort() 
            fleet = 0
            prev_time = 0
            while cars:
                p,t = cars.pop()
                if t > prev_time:
                    fleet +=1
                    prev_time = t  #update only if found greater time
            return fleet

| **Solution, initial**
| Keys:
| -sort input in reverse
| -work with values in stack[-1], stack[-2]. Put only time to stack.
 
| Steps
| -make pairs (position, speed)
| -sort pairs in reverse (meaning it will sort on position. Greatest..Smallest)
| NOTE: Greater position means the car is Closer to target ==> 4 is closer than 2.
| -loop our sorted pairs
| -On stack we will be putting "Time to get to the target"
| -The lookup stack[-1] and stack[-2] checks if:
| the Further away car (stack[-1]) would get to target Faster than the closer car (stack[-2]). 
| If yes, we can join that faster car to the fleet of the slower car: stack.pop()

Solution [:ref:`10 <ref-label>`] ::

    def carFleet(target: int, position: List[int], speed: List[int]) -> int:
        pair = [(p, s) for p, s in zip(position, speed)]
        pair.sort(reverse=True)
        stack = []                                      #[time1, time2..]
        for p, s in pair:  # Reverse Sorted Order
            stack.append((target - p) / s)              #**
            if len(stack) >= 2 and stack[-1] <= stack[-2]:
                stack.pop()
        return len(stack)

#** - 
Unlike many algs where we put to stack as last step, here we first put to stack,
and work with values stack[-1], stack[-2], because otherwise we don't check if
to merge the last car into a fleet with previous cars. 

C++
^^^^^^^^^^^^
| My V (LC accepted 18, 5)
| (Making a separate vector of vectors to store [[position, time]] is space costly, but helps with the clarity.)

.. code-block:: cpp

    class Solution {
    public:
        int carFleet(int target, vector<int>& position, vector<int>& speed) {
            vector<vector<double>> cars;
            for(size_t i = 0; i != position.size(); ++i){
                double t = (target - position[i]) / static_cast<double>(speed[i]);
                cars.push_back({static_cast<double>(position[i]), t});
            }
            sort(cars.begin(), cars.end());
            int fleet = 0;
            double prev_time = 0;
            while (!cars.empty()){
                double t = cars.back()[1];
                cars.pop_back();
                if(t > prev_time){
                    ++fleet;
                    prev_time = t;
                }
            }
            return fleet;
            }
    };

180. (LC 84) Largest Rectangle in Histogram
-----------------------------------------------
`84. Largest Rectangle in Histogram <https://leetcode.com/problems/largest-rectangle-in-histogram/description/>`_
Hard

| **Keys:**
| -If current height is greater than previous, then previous can extend its width +1.
| ->So monotonic increasing stack.
| -Store in stack [index, height]
 
| **Details:**
| -After popping from stack, record the cur value not with its cur index, 
| but with index of the last popped greater value.

::

    # stk=[(1,0),(5,2),(6,3)], next v=2, after popping stk=[(1,0),(2,3)]
    #              ^

| Because the lesser value can start (extends) from the first greater value.
| -After the main alg, when cleaning out values from stack, our stack is mono increasing,
| means all items in stack can extend to the last index of heights.
| int index = heights.size();
| area = max(area, v * (index-ind));

**My V** (LC accepted 35, 30%) ::

    class Solution:
        def largestRectangleArea(self, heights: List[int]) -> int:
            ans = 0
            stack = [] #[index, height]
            for i, h in enumerate(heights):
                start = i
                while stack and stack[-1][1] > h:
                    index, height = stack.pop()
                    area = height * (i-index)
                    ans = max(ans, area)
                    start = index
                stack.append([start, h])

            for i, h in stack:
                area = h * (len(heights) - i)
                ans = max(ans, area)
            return ans

**My V2** (LC accepted 30, 42%) ::

    class Solution:
        def largestRectangleArea(self, heights: List[int]) -> int:
            stack = []  #[(v,i),..] mono increasing
            area = 0
            for i, n in enumerate(heights):
                if not stack or stack[-1][0] <= n:
                    stack.append((n,i))
                else:
                    while stack and stack[-1][0] > n:
                        v, ind = stack.pop()
                        area = max(area, (i-ind)*v)
                    stack.append((n, ind))  #append to stack with last popped index
            
            index=len(heights)
            while stack:
                v,ind = stack.pop()
                area = max(area, v*(index-ind))
            return area

**Solution 1** [:ref:`10 <ref-label>`] ::

    class Solution:
        def largestRectangleArea(self, heights: List[int]) -> int:
            maxArea = 0
            stack = []  # pair: (index, height)

            for i, h in enumerate(heights):
                start = i                         #**1
                while stack and stack[-1][1] > h:
                    index, height = stack.pop()
                    maxArea = max(maxArea, height * (i - index))
                    start = index                 #**1
                stack.append((start, h))

            for i, h in stack:                   #**2
                maxArea = max(maxArea, h * (len(heights) - i))
            return maxArea

| #**1
| E.g.
| heights = [5,6,2]
| stack = [(1,5),(2,6)]
| Looking at the height 2.
| stack[-1][1]=5 > 2. We have to pop taller rects from the stack.
| Pop, calculate area.
| Then we record to stack not (2,2), but (0,2). 
| 0 being the ==> index of the last popped rect.
| ==> Because rect 2 can extend backwards all the way to index 0 here.
 
| Patterns:
| - heights = [1,2]
| Looking at 1, the next height is greater, means we can extend height=1 to the right.
| We put (i, h) to stack
|  ->2
| 1  2
 
| - heights = [2,1]
| Looking at 2, the next height is smaller, means we cannot extend height=2 to the right.
| 2->
| 2  1
| Then we pop all taller rects from stack. Record area for each.
| NOTE. The WIDTH = i of encountered shorter rect - index of popped rect 
| (See #**)
 
| #**2
| If there was only an upward trend in rects, then the stack will be full. 

::
 
    #     3
    #   2 3
    # 1 2 3

E.g.:
Area of rect 1 = 1 * ((len=3) - index=0) = 3

C++
^^^^^^^^^

.. code-block:: cpp

    //My V (LC accepted 65, 55%)
    class Solution {
    public:
        int largestRectangleArea(vector<int>& heights) {
            stack<pair<int,int>> stk;  //{{val, index},..}
            int area {0};
            for(int i=0; i!=heights.size(); ++i){
                int n = heights.at(i);
                if(stk.empty() || stk.top().first <= n)
                    stk.push({n,i});
                else{
                    int ind{0};  //have to declare not in while, otherwise stays in while locally
                    while(!stk.empty() && stk.top().first > n){
                        ind = stk.top().second;
                        int v = stk.top().first;
                        stk.pop();
                        area = max(area, v * (i-ind));
                    }
                    stk.push({n, ind});
                }
            }
            int index = heights.size();
            while(!stk.empty()){
                int ind = stk.top().second, v = stk.top().first;
                stk.pop();
                area = max(area, v * (index-ind));
            }
            return area;
        }
    };









