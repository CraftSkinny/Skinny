#hashed cells corrected (values are active if and only if at least diff or value is active, when both are inactive, value is not active)
#corrected the Kin and Kout
#improved version of key recovery, guessing the internal state in effective cases
from gurobipy import *
DistRound=9
PreRound=6
PostRound=6
marker = 0
filename_model = "SKI_"+str(PreRound)+"_"+str(DistRound)+"_"+str(PostRound)+".lp"
fileobj = open(filename_model, "w")
fileobj.close()
class SKINNY:
	MC = [[1,0,1,1],[1,0,0,0],[0,1,1,0],[1,0,1,0]]
	Inv_MC = [[0,1,0,0],[0,1,1,1],[0,1,0,1],[1,0,0,1]] #inverse of MC
	def column(A, j):
	    return [A[j], A[j+4], A[j+8], A[j+12]]

	def ShiftRow(A):
	    return [A[0], A[1], A[2], A[3],\
		    A[7], A[4], A[5], A[6],\
		    A[10],A[11],A[8], A[9],\
		    A[13],A[14],A[15],A[12]]
		    
class Truncated:
	S_T = [(0, 1, 2, 1, -1, 1, -1, 0, -4, -2, 0), (0, 0, -2, -1, 1, 0, 1, 1, 2, 1, 0), (0, -1, 0, 0, 1, 0, 1, -1, 2, 1, 0), (0, -1, 1, 0, 0, -1, 1, 1, 1, 0, 0), (0, 0, 0, 1, -1, 0, -2, -1, -4, -2, 4), (0, 1, 1, 0, 0, 0, -1, 0, -2, 0, 0), (0, 0, -2, 0, 0, 0, 1, 1, 2, 1, 0), (0, 1, -1, 0, 0, 0, 1, 0, 0, 0, 0),(0, 0, -2, 0, 1, 0, 1, 0, 2, 1, 0), (0, 0, 1, 1, -1, 1, -2, 0, -4, -2, 2), (0, 1, 2, 0, -2, 1, -1, -1, -4, -2, 2), (0, -1, 0, -1, 1, -1, 1, 1, 2, 1, 0)]
	#4p0 , 8p1:
	#S_T_Back = [(0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0),(1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0),(0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0),(0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0),(0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 3),(0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1),(1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0),(0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0),(1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0),(0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0),(0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1),(0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0),(0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0),(1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1),(0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0),(0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0),(0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0),(0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0),(0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 2),(0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0)]
	#8p0 , 4p1
	S_T_Back = [(0, 0, -1, 0, 0, 1, 1, 0, 0, 0, 0),(1, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0),(0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0),(0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0),(0, 0, 0, 0, 0, -1, -1, -1, 0, -1, 3),(0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 1),(-1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0),(0, 0, 1, 0, 0, -1, 1, 0, 0, 0, 0),(1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0),(0, 0, 1, 0, 0, 1, -1, 0, 0, 0, 0),(0, 0, 0, -1, 0, 1, 0, 1, 0, -1, 1),(0, 0, 0, 1, 0, 0, 1, 0, 0, -1, 0),(0, 0, 0, -1, 1, 0, 1, 0, 0, 0, 0),(1, 0, 0, 0, 0, -1, 0, 1, 0, -1, 1),(0, 0, 0, 0, 0, 1, -1, 0, 1, 1, 0),(0, 0, 0, 1, 1, 0, -1, 0, 0, 0, 0),(0, 0, 0, 0, -1, 0, 1, 0, 1, 1, 0),(0, 0, 0, -1, 0, 0, 0, 1, 1, 1, 0),(0, 0, 0, 0, 0, -1, -1, 0, -1, 0, 2),(0, 0, 0, 0, 0, 0, 1, 1, 0, -1, 0)] 

	def genEncSubjectionAtRound(r):

		eqn = []

		inX = ['x_InMC_' + str(r) + '_' + str(i) for i in range(0,16)]
		outY = ['x_InSB_' + str(r+1) + '_' + str(i) for i in range(0,16)]

		eqn = eqn + Truncated.mixColumnAndShiftRow(inX,outY,Truncated.S_T)

		return eqn

	def genEncSubjectionAtRoundPre(r):

		eqn = []

		outY = ['x_InMC_' + str(r) + '_' + str(i) for i in range(0,16)]
		inX = ['x_InSB_' + str(r+1) + '_' + str(i) for i in range(0,16)]

		eqn = eqn + Truncated.mixColumnAndShiftRowPre(inX,outY,Truncated.S_T_Back)

		return eqn


	def genEncSubjection(totalRound,PreRound,PostRound):
		eqn = []

		for i in range(PreRound):
			eqn = eqn + Truncated.genEncSubjectionAtRoundPre(i)

		for i in range(PreRound, totalRound + PreRound+PostRound):
			eqn = eqn + Truncated.genEncSubjectionAtRound(i)

		return eqn


	
	def mixColumnSubjection(inX,outY,ineq):
		assert(len(inX) == len(outY) and len(inX) == 4)
		global marker
		eqn = []
		T = ineq

		for t in T:
			eqn.append((str(t[0]) + ' ' + inX[0] + ' + ' + str(t[1]) + ' ' + inX[1] + ' + ' + str(t[2]) + ' ' + inX[2] + ' + ' + str(t[3]) + ' ' + inX[3] + ' + ' + str(t[4]) + ' ' + outY[0] + ' + ' + str(t[5]) + ' ' + outY[1] + ' + ' + str(t[6]) + ' ' + outY[2] + ' + ' + str(t[7]) + ' ' + outY[3] + ' + ' +  str(t[8]) + ' ' + 'p0' + '_' + str(marker) + ' + ' + str(t[9]) + ' ' + 'p1' +  '_' + str(marker) +  ' >= ' + str(-t[10])).replace('+ -','- '))

		eqn.append(str(1) + ' ' + inX[0] + ' - ' + str(1) + ' ' + outY[1] + ' = 0')
		marker = marker +1
		return eqn

	def mixColumnSubjectionPre(inX,outY,ineq):
		assert(len(inX) == len(outY) and len(inX) == 4)
		global marker
		eqn = []
		T = ineq

		for t in T:
			eqn.append((str(t[0]) + ' ' + inX[0] + ' + ' + str(t[1]) + ' ' + inX[1] + ' + ' + str(t[2]) + ' ' + inX[2] + ' + ' + str(t[3]) + ' ' + inX[3] + ' + ' + str(t[4]) + ' ' + outY[0] + ' + ' + str(t[5]) + ' ' + outY[1] + ' + ' + str(t[6]) + ' ' + outY[2] + ' + ' + str(t[7]) + ' ' + outY[3] + ' + ' +  str(t[8]) + ' ' + 'p0' + '_' + str(marker) + ' + ' + str(t[9]) + ' ' + 'p1' +  '_' + str(marker) +  ' >= ' + str(-t[10])).replace('+ -','- '))

		#eqn.append(str(1) + ' ' + inX[0] + ' - ' + str(1) + ' ' + outY[1] + ' = 0')
		marker = marker +1
		return eqn
	    
	def mixColumnAndShiftRow(inX,outY,ineq):

		eqn = []
		
		inX1 = [inX[0],inX[4],inX[8],inX[12]]
		outY1 = [outY[0],outY[4],outY[8],outY[12]]
		inX2 = [inX[1],inX[5],inX[9],inX[13]]
		outY2 = [outY[1],outY[5],outY[9],outY[13]]
		inX3 = [inX[2],inX[6],inX[10],inX[14]]
		outY3 = [outY[2],outY[6],outY[10],outY[14]]
		inX4 = [inX[3],inX[7],inX[11],inX[15]]
		outY4 = [outY[3],outY[7],outY[11],outY[15]]
        		

		eqn = eqn + Truncated.mixColumnSubjection(inX1,outY1,ineq)
		eqn = eqn + Truncated.mixColumnSubjection(inX2,outY2,ineq)
		eqn = eqn + Truncated.mixColumnSubjection(inX3,outY3,ineq)
		eqn = eqn + Truncated.mixColumnSubjection(inX4,outY4,ineq)
    
		return eqn

	def mixColumnAndShiftRowPre(inX,outY,ineq):

		eqn = []
		
		inX1 = [inX[0],inX[4],inX[8],inX[12]]
		outY1 = [outY[0],outY[4],outY[8],outY[12]]
		inX2 = [inX[1],inX[5],inX[9],inX[13]]
		outY2 = [outY[1],outY[5],outY[9],outY[13]]
		inX3 = [inX[2],inX[6],inX[10],inX[14]]
		outY3 = [outY[2],outY[6],outY[10],outY[14]]
		inX4 = [inX[3],inX[7],inX[11],inX[15]]
		outY4 = [outY[3],outY[7],outY[11],outY[15]]
        		

		eqn = eqn + Truncated.mixColumnSubjectionPre(inX1,outY1,ineq)
		eqn = eqn + Truncated.mixColumnSubjectionPre(inX2,outY2,ineq)
		eqn = eqn + Truncated.mixColumnSubjectionPre(inX3,outY3,ineq)
		eqn = eqn + Truncated.mixColumnSubjectionPre(inX4,outY4,ineq)
    
		return eqn
	    
class BasicTools:
	def transpose(M):
		m = len(M)
		n = len(M[0])

		Mt = []
		for i in range(0, n):
			row = [M[k][i] for k in range(0, m)]
			Mt.append(row)

		return Mt
		
	def VarGen(s,n):
	    return [str(s) + '_' + str(n) + '_' + str(i) for i in range(0,16)]
	    
	def plusTerm(in_vars):
		t = ''
		for v in in_vars:
		    t = t + v + ' + '

		return t[0:-3]

	def MinusTerm(in_vars):
		t = ''
		for v in in_vars:
		    t = t + v + ' - '

		return t[0:-3]	    
		
	def equalConstraints(x, y):
		assert len(x) == len(y)
		c = []
		for i in range(0, len(x)):
	    		c = c + [x[i] + ' - ' + y[i] + ' = 0']
		return c
		
	def greaterConstraints(x, y):
		assert len(x) == len(y)
		c = []
		for i in range(0, len(x)):
	    		c = c + [x[i] + ' - ' + y[i] + ' >= 0']
		return c
		
	def greaterConstraints_xyz(x, y, z):
		assert len(x) == len(y)
		c = []
		for i in range(0, len(x)):
	    		c = c + [x[i] + ' + ' + y[i] + ' - ' + z[i] + ' >= 0']
		return c			    
				
	def getVariables(C):
		V = set([])
		temp = C.strip()
		        
		temp = temp.replace('+', ' ')
		temp = temp.replace('-', ' ')
		temp = temp.replace('>=', ' ')
		temp = temp.replace('<=', ' ')
		temp = temp.replace('=', ' ')
		temp = temp.split()
		for v in temp :
		        if not v.isdecimal():
		                V.add(v)
		return V

class Extension:		
	def ForwardDiff_LinearLayer(M, V_in, V_out):
		assert len(M[0]) == len(V_in)
		assert len(M) == len(V_out)


		m = len(M)
		n = len(M[0])

		constr = []
		for i in range(0, m):
		    s = sum(M[i]) # the number of 1s in row i
		    terms = [V_in[j] for j in range(0, n) if M[i][j] == 1]
		    constr = constr + [str(s) + ' ' + V_out[i] + ' - ' + ' ' + BasicTools.MinusTerm(terms) + ' >= 0']
		    constr = constr + [BasicTools.plusTerm(terms) + ' - ' + V_out[i] + ' >= 0']

		return constr
	def LinearLayer(M, V_in, V_out):
		assert len(M[0]) == len(V_in)
		assert len(M) == len(V_out)


		m = len(M)
		n = len(M[0])

		constr = []
		for i in range(0, m):
		    s = sum(M[i]) # the number of 1s in row i
		    terms = [V_in[j] for j in range(0, n) if M[i][j] == 1]
		    constr = constr + [str(s) + ' ' + V_out[i] + ' - ' + ' ' + BasicTools.MinusTerm(terms) + ' >= 0']
		    constr = constr + [BasicTools.plusTerm(terms) + ' - ' + V_out[i] + ' >= 0']

		return constr	

	def McRelImprovementForward(inX,W):
		T=[("x2+y3'"),("x0+y1'"),("x2+y1'"),("x1+y2'"),("x3+y0'"),("x3'+y0"),("x1'+y2"),("y2+y3'"),("x0'+x2'+y1+y2'"),("y0+y3'"),("x2'+y0'+y2'+y3"),("y1'+y2")]
		eqn = []

		for t in T:
			cnt=0
			eq=""
			if ("x0'" in t):
				cnt+=1
				eq=eq+("- " + inX[0])
				#eq=eq+("1 - " + inX[0])
			elif ("x0" in t):
				eq=eq+(inX[0])
			if ("x1'" in t):
				cnt+=1
				eq=eq+(" - " + inX[1])
				#eq=eq+(" + 1 - " + inX[1])
			elif ("x1" in t):
				eq=eq+(" + " + inX[1])
			if ("x2'" in t):
				cnt+=1
				eq=eq+(" - " + inX[2])
				#eq=eq+(" + 1 - " + inX[2])
			elif ("x2" in t):
				eq=eq+(" + " + inX[2])
			if ("x3'" in t):
				cnt+=1
				eq=eq+(" - " + inX[3])
				#eq=eq+(" + 1 - " + inX[3])
			elif ("x3" in t):
				eq=eq+(" + " + inX[3])
			if ("y0'" in t):
				cnt+=1
				eq=eq+(" - " + W[0])
			elif ("y0" in t):
				eq=eq+(" + " + W[0])
			if ("y1'" in t):
				cnt+=1
				eq=eq+(" - " + W[1])
			elif ("y1" in t):
				eq=eq+(" + " + W[1])
			if ("y2'" in t):
				cnt+=1
				eq=eq+(" - " + W[2])
			elif ("y2" in t):
				eq=eq+(" + " + W[2])
			if ("y3'" in t):
				cnt+=1
				eq=eq+(" - " + W[3])
			elif ("y3" in t):
				eq=eq+(" + " + W[3])
			eq=eq + " >= " + str(-cnt+1)
			eqn.append(eq)
		return eqn
		
	def McRelImprovementBackward(inX,W):
		T = [("x3+y2'"),("x1+y0'"),("x3+y0'"),("x0+y3'"),("x2+y1'"),("x2'+y1"),("x0'+y3"),("y2'+y3"),("x1'+x3'+y0+y3'"),("y1+y2'"),("x3'+y1'+y2+y3'"),("y0'+y3")]
		eqn = []

		for t in T:
			cnt=0
			eq=""
			if ("x0'" in t):
				cnt+=1
				eq=eq+("- " + inX[0])
				#eq=eq+("1 - " + inX[0])
			elif ("x0" in t):
				eq=eq+(inX[0])
			if ("x1'" in t):
				cnt+=1
				eq=eq+(" - " + inX[1])
				#eq=eq+(" + 1 - " + inX[1])
			elif ("x1" in t):
				eq=eq+(" + " + inX[1])
			if ("x2'" in t):
				cnt+=1
				eq=eq+(" - " + inX[2])
				#eq=eq+(" + 1 - " + inX[2])
			elif ("x2" in t):
				eq=eq+(" + " + inX[2])
			if ("x3'" in t):
				cnt+=1
				eq=eq+(" - " + inX[3])
				#eq=eq+(" + 1 - " + inX[3])
			elif ("x3" in t):
				eq=eq+(" + " + inX[3])
			if ("y0'" in t):
				cnt+=1
				eq=eq+(" - " + W[0])
			elif ("y0" in t):
				eq=eq+(" + " + W[0])
			if ("y1'" in t):
				cnt+=1
				eq=eq+(" - " + W[1])
			elif ("y1" in t):
				eq=eq+(" + " + W[1])
			if ("y2'" in t):
				cnt+=1
				eq=eq+(" - " + W[2])
			elif ("y2" in t):
				eq=eq+(" + " + W[2])
			if ("y3'" in t):
				cnt+=1
				eq=eq+(" - " + W[3])
			elif ("y3" in t):
				eq=eq+(" + " + W[3])
			eq=eq + " >= " + str(-cnt+1)
			eqn.append(eq)
		return eqn
			
	def Determination_Decision(M, V_in, V_out , t_variable):
		assert len(M[0]) == len(V_in)
		assert len(M) == len(V_out)
		m = len(M)
		n = len(M[0])
		constr = []

		for j in range(0, 4):#This Forloop Indicates That How Yi and Ti(variable) Related (Yi >= Ti).Yi is output of mixcolumn matrix, and for each active nibble we consider T variable
			constr=constr+[V_out[j] + ' - ' + t_variable[j] + ' >= 0' ]
		if True:
			for i in range(0, m):#This Forloop Shows That If Yi are active and Ti variables is non-active
				s = sum(M[i]) # the number of 1s in row i
				terms1=[V_in[j] for j in range(0, n) if M[i][j] == 1]
				constr = constr + [BasicTools.plusTerm(terms1) + ' - ' + str(s) + ' ' + V_out[i] + ' + ' + str(s) + ' ' + t_variable[i] + ' >= 0']
		if True:

			for j in range(0, m):#This Forloop shows that if Ti variable are now active and we want to describe the constraints
				s = sum(M[k][j] for k in range(0,len(M))) # the number of 1s in column j
				terms1=[t_variable[i] for i in range(0, n) if M[i][j] == 1]
				terms2 = [ V_out[i] for i in range(0, n) if M[i][j] == 1]
				constr = constr + [BasicTools.plusTerm(terms1) + ' + ' + ' ' + V_in[j] + ' ' + ' - ' +  (BasicTools.MinusTerm(terms2)) + ' <= 0']#original       
		return constr
        	
	def BackwardDet_LinearLayer(M, V_in, V_out):
		"""
		>>> M = [[1,0,1,1],[1,0,0,0],[0,1,1,0],[1,0,1,0]]
		>>> a = ['a0', 'a1', 'a2', 'a3']
		>>> b = ['b0', 'b1', 'b2', 'b3']
		>>> MITMConstraints.BackwardDet_LinearLayer(M, a, b)
		['3 a0 -  b0 - b1 - b3 >= 0',
		 'b0 + b1 + b3 - a0 >= 0',
		 '1 a1 -  b2 >= 0',
		 'b2 - a1 >= 0',
		 '3 a2 -  b0 - b2 - b3 >= 0',
		 'b0 + b2 + b3 - a2 >= 0',
		 '1 a3 -  b0 >= 0',
		 'b0 - a3 >= 0']
		>>>
		>>>
		"""
		return Extension.ForwardDiff_LinearLayer(BasicTools.transpose(M), V_out, V_in)	
	def genConstraints_backwardkeyrecovery(r): 
		Input_round_diff = BasicTools.VarGen("x_InSB",r)
		Input_MC_diff = BasicTools.VarGen("x_InMC",r)
		Output_round_diff = BasicTools.VarGen("x_InSB",r+1)
		Input_round_val = BasicTools.VarGen("y_InSB",r)
		Input_MC_val = BasicTools.VarGen("y_InMC",r)
		Output_round_val = BasicTools.VarGen("y_InSB",r+1)
		Input_round_h = BasicTools.VarGen("h_InSB",r)
		Input_MC_h = BasicTools.VarGen("h_InMC",r)
		Output_round_h = BasicTools.VarGen("h_InSB",r+1)
		DeterminationVar = BasicTools.VarGen("t",r)	     
		Constr = []
		#for j in range(4):
		#	Constr = Constr + Extension.LinearLayer(SKINNY.Inv_MC, SKINNY.column(Output_round_diff,j), SKINNY.column(Input_MC_diff,j))
		Constr = Constr + BasicTools.equalConstraints(SKINNY.ShiftRow(Input_round_diff), Input_MC_diff)
		for j in range(4):
			#Constr = Constr + Extension.LinearLayer(BasicTools.transpose(SKINNY.MC), SKINNY.column(Output_round_val,j), SKINNY.column(Input_MC_val,j))
			Constr = Constr + Extension.Determination_Decision(SKINNY.MC, SKINNY.column(Input_MC_val,j), SKINNY.column(Output_round_val,j), SKINNY.column(DeterminationVar,j))
		Constr = Constr + BasicTools.equalConstraints(SKINNY.ShiftRow(Input_round_h), Input_MC_val)
		#Constr = Constr + BasicTools.greaterConstraints(Output_round_val, Output_round_diff)
		Constr = Constr + BasicTools.greaterConstraints(Output_round_val, Output_round_diff)
		Constr = Constr + BasicTools.greaterConstraints(Output_round_val, Output_round_h)
		Constr = Constr + BasicTools.greaterConstraints_xyz(Output_round_h,Output_round_diff,Output_round_val)		
		return Constr		
	def genConstraints_backwardkeyrecoveryLastR(r): 
		Input_round_diff = BasicTools.VarGen("x_InSB",r)
		Input_MC_diff = BasicTools.VarGen("x_InMC",r)
		Output_round_diff = BasicTools.VarGen("x_InSB",r+1)
		Input_round_val = BasicTools.VarGen("y_InSB",r)
		Input_MC_val = BasicTools.VarGen("y_InMC",r)
		Output_round_val = BasicTools.VarGen("y_InSB",r+1)     
		Input_MC_McRel = BasicTools.VarGen("w",r)
		Constr = []
		for j in range(4):
			#Constr = Constr + Extension.LinearLayer(SKINNY.Inv_MC, SKINNY.column(Output_round_diff,j), SKINNY.column(Input_MC_diff,j))		
			Constr = Constr + Extension.McRelImprovementBackward(SKINNY.column(Output_round_diff,j), SKINNY.column(Input_MC_McRel,j))
		return Constr
				
	def genConstraints_forwardkeyrecovery(r): 
		Input_round_diff = BasicTools.VarGen("x_InSB",r)
		Input_MC_diff = BasicTools.VarGen("x_InMC",r)
		Output_round_diff = BasicTools.VarGen("x_InSB",r+1)
		Input_round_val = BasicTools.VarGen("y_InSB",r)
		Input_MC_val = BasicTools.VarGen("y_InMC",r)
		Output_round_val = BasicTools.VarGen("y_InSB",r+1)
		Input_round_h = BasicTools.VarGen("h_InSB",r)
		Input_MC_h = BasicTools.VarGen("h_InMC",r)
		Output_round_h = BasicTools.VarGen("h_InSB",r+1)
		DeterminationVar = BasicTools.VarGen("t",r)
	     
		Constr = []
		#for j in range(4):
		#	Constr = Constr + Extension.LinearLayer(SKINNY.MC, SKINNY.column(Input_MC_diff,j), SKINNY.column(Output_round_diff,j))
		Constr = Constr + BasicTools.equalConstraints(SKINNY.ShiftRow(Input_round_diff), Input_MC_diff)
		for j in range(4):
			#Constr = Constr + Extension.LinearLayer(BasicTools.transpose(SKINNY.Inv_MC), SKINNY.column(Input_MC_h,j), SKINNY.column(Output_round_h,j))
			Constr = Constr + Extension.Determination_Decision(SKINNY.Inv_MC, SKINNY.column(Output_round_h,j), SKINNY.column(Input_MC_h,j), SKINNY.column(DeterminationVar,j))
		Constr = Constr + BasicTools.equalConstraints(SKINNY.ShiftRow(Input_round_val), Input_MC_h)
		Constr = Constr + BasicTools.greaterConstraints(Output_round_val, Output_round_diff)
		Constr = Constr + BasicTools.greaterConstraints(Output_round_val, Output_round_h)
		Constr = Constr + BasicTools.greaterConstraints_xyz(Output_round_h,Output_round_diff,Output_round_val)
		
		return Constr
	def genConstraints_forwardkeyrecoveryFirstR(r): 
		Input_round_diff = BasicTools.VarGen("x_InSB",r)
		Input_MC_diff = BasicTools.VarGen("x_InMC",r)
		Output_round_diff = BasicTools.VarGen("x_InSB",r+1)
		Input_round_val = BasicTools.VarGen("y_InSB",r)
		Input_MC_val = BasicTools.VarGen("y_InMC",r)
		Output_round_val = BasicTools.VarGen("y_InSB",r+1)
		Output_round_McRel = BasicTools.VarGen("w",r+1)
	     
		Constr = []
		for j in range(4):
			#Constr = Constr + Extension.LinearLayer(SKINNY.MC, SKINNY.column(Input_MC_diff,j), SKINNY.column(Output_round_diff,j))	
			Constr = Constr + Extension.McRelImprovementForward(SKINNY.column(Input_MC_diff,j), SKINNY.column(Output_round_McRel,j))		
		return Constr		
        	    		    
if __name__ == '__main__':
	const=[]
	fileobj = open(filename_model, "w")	
	#----------------------------------------2
	fileobj.write("Minimize\n")
	fileobj.write("Dummy\n")
	fileobj.write("Subject to\n")
	#fileobj.write("p1_72 = 1\n")
	fileobj.write("Dummy - ")
	fileobj.write(' - '.join( ['80 p0' + '_' + str(i) for i in range(PreRound*4,(PreRound+DistRound)*4)] + ['40 p1' + '_' + str(i) for i in range(PreRound*4,(PreRound+DistRound)*4)]))#p
	fileobj.write(" - ")
	fileobj.write(' - '.join( ['80 p0' + '_' + str(i) for i in range((PreRound+DistRound)*4,(PreRound+DistRound+PostRound)*4)] + ['40 p1' + '_' + str(i) for i in range((PreRound+DistRound)*4,(PreRound+DistRound+PostRound)*4)]))#p_out
	fileobj.write(" - ")
	fileobj.write(' - '.join( ['40 u_InSB' + '_' + str(i) + '_' + str(j) for i in range(0,PreRound-1) for j in range(0,4)]))#k_in
	fileobj.write(" - ")
	fileobj.write(' - '.join( ['40 u_InSB' + '_' + str(i) + '_' + str(j) for i in range(0,PreRound-1) for j in range(8,12)]))#k_in
	fileobj.write(" - ")
	fileobj.write(' - '.join( ['41 t' + '_' + str(i) + '_' + str(j) for i in range(0,PreRound-1) for j in range(0,16)]))#t in k_in
	fileobj.write(" - ")
	fileobj.write(' - '.join( ['t' + '_' + str(i) + '_' + str(j) for i in range(PreRound+DistRound+1,PreRound+DistRound+PostRound) for j in range(0,16)]))#t in k_out
	fileobj.write(" >= 0 \n")
	fileobj.write(' + '.join( ['x_InMC_' + str(PreRound) + '_' + str(j) for j in range(0,16)]))#delta_in
	fileobj.write(" >= 1\n")
	####################################
	fileobj.write("Dummy - ")
	fileobj.write(' - '.join( ['80 p0' + '_' + str(i) for i in range(PreRound*4,(PreRound+DistRound)*4)] + ['40 p1' + '_' + str(i) for i in range(PreRound*4,(PreRound+DistRound)*4)]))#p
	fileobj.write(" - ")
	fileobj.write(' - '.join( ['80 p0' + '_' + str(i) for i in range(0,PreRound*4)] + ['40 p1' + '_' + str(i) for i in range(0,PreRound*4)]))#p_in
	fileobj.write(" - ")
	fileobj.write(' - '.join( ['40 z_InSB' + '_' + str(i) + '_' + str(j) for i in range(PreRound+DistRound+1,PreRound+DistRound+PostRound) for j in range(0,8)]))#k_out
	fileobj.write(" - ")
	fileobj.write(' - '.join( ['41 t' + '_' + str(i) + '_' + str(j) for i in range(PreRound+DistRound+1,PreRound+DistRound+PostRound) for j in range(0,16)]))#t in k_out
	fileobj.write(" - ")
	fileobj.write(' - '.join( ['t' + '_' + str(i) + '_' + str(j) for i in range(0,PreRound-1) for j in range(0,16)]))#t in k_in
	fileobj.write(" - ")
	fileobj.write(' - '.join( ['40 x_InSB_' + str(PreRound+DistRound) + '_' + str(j) for j in range(0,16)]))#delta_out
	fileobj.write(" + ")
	fileobj.write(' + '.join( ['40 x_InMC_' + str(PreRound) + '_' + str(j) for j in range(0,16)]))#delta_in
	fileobj.write(" >= 0 \n")

	####################################
	fileobj.write("Dummy - ")
	fileobj.write(' - '.join( ['80 p0' + '_' + str(i) for i in range(PreRound*4,(PreRound+DistRound)*4)] + ['40 p1' + '_' + str(i) for i in range(PreRound*4,(PreRound+DistRound)*4)]))#p
	fileobj.write(" - ")
	fileobj.write(' - '.join( ['80 p0' + '_' + str(i) for i in range(0,PreRound*4)] + ['40 p1' + '_' + str(i) for i in range(0,PreRound*4)]))#p_in
	fileobj.write(" - ")
	fileobj.write(' - '.join( ['80 p0' + '_' + str(i) for i in range((PreRound+DistRound)*4,(PreRound+DistRound+PostRound)*4)] + ['40 p1' + '_' + str(i) for i in range((PreRound+DistRound)*4,(PreRound+DistRound+PostRound)*4)]))#p_out
	fileobj.write(" - ")
	fileobj.write(' - '.join( ['40 t' + '_' + str(i) + '_' + str(j) for i in range(PreRound+DistRound+1,PreRound+DistRound+PostRound) for j in range(0,16)]))#t in k_out
	fileobj.write(" - ")
	fileobj.write(' - '.join( ['40 t' + '_' + str(i) + '_' + str(j) for i in range(0,PreRound-1) for j in range(0,16)]))#t in k_in
	fileobj.write(" - ")
	fileobj.write(' - '.join( ['40 x_InSB_' + str(PreRound+DistRound) + '_' + str(j) for j in range(0,16)]))#delta_out
	fileobj.write(" >= 1280 \n")

	
	fileobj.write("Dummy2 - ")
	fileobj.write(' - '.join( ['8 p0' + '_' + str(i) for i in range(PreRound*4,(PreRound+DistRound)*4)] + ['4 p1' + '_' + str(i) for i in range(PreRound*4,(PreRound+DistRound)*4)]))#p
	fileobj.write(" = 0 \n")
	
	fileobj.write("Dummy3 - ")
	fileobj.write(' - '.join( ['4 x_InSB_' + str(PreRound+DistRound) + '_' + str(j) for j in range(0,16)]))#delta_out
	fileobj.write(" = 0 \n")
	
	fileobj.write("Dummy4 - ")
	fileobj.write(' - '.join( ['4 x_InMC_' + str(PreRound) + '_' + str(j) for j in range(0,16)]))#delta_in
	fileobj.write(" = 0 \n")
	
	fileobj.write("Dummy5 - ")
	fileobj.write(' - '.join( ['4 z_InSB' + '_' + str(i) + '_' + str(j) for i in range(PreRound+DistRound+1,PreRound+DistRound+PostRound) for j in range(0,8)]))#k_out
	fileobj.write(" - ")
	fileobj.write(' - '.join( ['4 t' + '_' + str(i) + '_' + str(j) for i in range(PreRound+DistRound+1,PreRound+DistRound+PostRound) for j in range(0,16)]))#t in k_out
	fileobj.write(" = 0 \n")
	
	fileobj.write("Dummy6 - ")
	fileobj.write(' - '.join( ['4 u_InSB' + '_' + str(i) + '_' + str(j) for i in range(0,PreRound-1) for j in range(0,4)]))#k_in
	fileobj.write(" - ")
	fileobj.write(' - '.join( ['4 u_InSB' + '_' + str(i) + '_' + str(j) for i in range(0,PreRound-1) for j in range(8,12)]))#k_in
	fileobj.write(" - ")
	fileobj.write(' - '.join( ['4 t' + '_' + str(i) + '_' + str(j) for i in range(0,PreRound-1) for j in range(0,16)]))#t in k_in
	fileobj.write(" = 0 \n")

	fileobj.write("Dummy7 - ")
	fileobj.write(' - '.join( ['8 p0' + '_' + str(i) for i in range(0,PreRound*4)] + ['4 p1' + '_' + str(i) for i in range(0,PreRound*4)]))#p_in
	fileobj.write(" = 0 \n")
	
	#fileobj.write("Dummy8 > 1 \n")
	fileobj.write("Dummy8 - ")
	fileobj.write(' - '.join( ['8 p0' + '_' + str(i) for i in range((PreRound+DistRound)*4,(PreRound+DistRound+PostRound)*4)] + ['4 p1' + '_' + str(i) for i in range((PreRound+DistRound)*4,(PreRound+DistRound+PostRound)*4)]))#p_out
	fileobj.write(" = 0 \n")
	
	fileobj.write("Dummy9 - ")
	fileobj.write(' - '.join( ['8 p0' + '_' + str(i) for i in range(0,(PreRound+DistRound+PostRound)*4)] + ['4 p1' + '_' + str(i) for i in range(0,(PreRound+DistRound+PostRound)*4)]))#p
	fileobj.write(" + ")
	fileobj.write(' + '.join( ['4 x_InMC_' + str(PreRound) + '_' + str(j) for j in range(0,16)]))#delta_in
	fileobj.write(" = 0 \n") #data complexity p-delta_in<n=64	
	####################################	
	fileobj.write(' + '.join( ['8 p0' + '_' + str(i) for i in range(0,(PreRound+DistRound+PostRound)*4)] + ['4 p1' + '_' + str(i) for i in range(0,(PreRound+DistRound+PostRound)*4)]))#p
	fileobj.write(" + ")
	fileobj.write(' + '.join( ['4 x_InMC_' + str(PreRound) + '_' + str(j) for j in range(0,16)]))
	fileobj.write(" <= 63\n")#data complexity p-delta_in<n=64
	####################################	
	fileobj.write(' + '.join( ['8 p0' + '_' + str(i) for i in range(PreRound*4,(PreRound+DistRound)*4)] + ['4 p1' + '_' + str(i) for i in range(PreRound*4,(PreRound+DistRound)*4)]))#p
	fileobj.write(" + ")
	fileobj.write(' + '.join( ['4 x_InSB_' + str(PreRound+DistRound) + '_' + str(j) for j in range(0,16)]))
	fileobj.write(" < 63\n")#prp
	const+=Truncated.genEncSubjection(DistRound+1,PreRound,PostRound)
	for i in range(0,PreRound+DistRound+PostRound):
		const+=BasicTools.equalConstraints(SKINNY.ShiftRow(BasicTools.VarGen("x_InSB",i)), BasicTools.VarGen("x_InMC",i))
		const+=BasicTools.greaterConstraints((BasicTools.VarGen("z_InSB",i)), BasicTools.VarGen("x_InSB",i))
		const+=BasicTools.greaterConstraints((BasicTools.VarGen("z_InSB",i)), BasicTools.VarGen("y_InSB",i))
					
	#const+=BasicTools.equalConstraints(BasicTools.VarGen("x_InSB",PreRound-1), BasicTools.VarGen("y_InSB",PreRound-1))
	c = []
	x=SKINNY.ShiftRow(BasicTools.VarGen("x_InSB",PreRound-1))
	y=SKINNY.ShiftRow(BasicTools.VarGen("y_InSB",PreRound-1))
	w=BasicTools.VarGen("w",PreRound-1)
	for i in range(0, 16):
    		c = c + [x[i] + ' - ' + w[i] + ' - ' + y[i] + ' = 0']
	const+=c
	
	for i in range(PreRound-1):
		const+=Extension.genConstraints_backwardkeyrecovery(i)
	const+=Extension.genConstraints_backwardkeyrecoveryLastR(PreRound-1)
	
	
	for i in range(1,PreRound):
		zprime=['zp_InSB_' + str(i) + '_' + str(j) for j in range(0,16)]
		z=['z_InSB_' + str(i) + '_' + str(j) for j in range(0,16)]
		t=['t_' + str(i-1) + '_' + str(j) for j in range(0,16)]
		for j in range(16):
			eqn=[]
			eqn.append(z[j] + ' - ' + t[j] + ' - ' + zprime[j] + ' = 0 ')
			const += eqn
			
	for i in range(1,PreRound):#constraints for equivalent Kin (redundancy)
		eqKin = BasicTools.VarGen("u_InSB",i-1)
		eqn=[]
		eqn.append(eqKin[0] + ' - zp_InSB_' + str(i) + '_0 >= 0')
		const += eqn
		eqn=[]
		eqn.append(eqKin[0] + ' - zp_InSB_' + str(i) + '_4 >= 0')
		const += eqn
		eqn=[]
		eqn.append(eqKin[0] + ' - zp_InSB_' + str(i) + '_12 >= 0')
		const += eqn
		eqn=[]
		eqn.append(eqKin[8] + ' - zp_InSB_' + str(i) + '_8 >= 0')
		const += eqn		
		eqn=[]
		eqn.append(eqKin[1] + ' - zp_InSB_' + str(i) + '_1 >= 0')
		const += eqn
		eqn=[]
		eqn.append(eqKin[1] + ' - zp_InSB_' + str(i) + '_5 >= 0')
		const += eqn
		eqn=[]
		eqn.append(eqKin[1] + ' - zp_InSB_' + str(i) + '_13 >= 0')
		const += eqn
		eqn=[]
		eqn.append(eqKin[9] + ' - zp_InSB_' + str(i) + '_9 >= 0')
		const += eqn		
		eqn=[]
		eqn.append(eqKin[2] + ' - zp_InSB_' + str(i) + '_2 >= 0')
		const += eqn
		eqn=[]
		eqn.append(eqKin[2] + ' - zp_InSB_' + str(i) + '_6 >= 0')
		const += eqn
		eqn=[]
		eqn.append(eqKin[2] + ' - zp_InSB_' + str(i) + '_14 >= 0')
		const += eqn
		eqn=[]
		eqn.append(eqKin[10] + ' - zp_InSB_' + str(i) + '_10 >= 0')
		const += eqn	
		eqn=[]
		eqn.append(eqKin[3] + ' - zp_InSB_' + str(i) + '_3 >= 0')
		const += eqn
		eqn=[]
		eqn.append(eqKin[3] + ' - zp_InSB_' + str(i) + '_7 >= 0')
		const += eqn
		eqn=[]
		eqn.append(eqKin[3] + ' - zp_InSB_' + str(i) + '_15 >= 0')
		const += eqn
		eqn=[]
		eqn.append(eqKin[11] + ' - zp_InSB_' + str(i) + '_11 >= 0')
		const += eqn	

	####################################
	#const+=BasicTools.equalConstraints(BasicTools.VarGen("x_InMC",PreRound+DistRound+1), BasicTools.VarGen("h_InMC",PreRound+DistRound+1))
	c = []
	x=BasicTools.VarGen("x_InMC",PreRound+DistRound+1)
	h=BasicTools.VarGen("h_InMC",PreRound+DistRound+1)
	w=SKINNY.ShiftRow(BasicTools.VarGen("w",PreRound+DistRound+1))
	for i in range(0, 16):
    		c = c + [x[i] + ' - ' + w[i] + ' - ' + h[i] + ' = 0']
	const+=c
	const+=Extension.genConstraints_forwardkeyrecoveryFirstR(PreRound+DistRound)
	for i in range(PreRound+DistRound+1,PreRound+DistRound+PostRound):
		const+=Extension.genConstraints_forwardkeyrecovery(i)
	####################################		
	for c in const:
		fileobj.write(str(c))
		fileobj.write("\n")
	fileobj.write("Binary\n")
	for c in const:
		Var = BasicTools.getVariables(c)
		#print(c)
		#print(Var)
		for v in Var:
			fileobj.write(v)
			fileobj.write("\n")
	fileobj.write("General\n")
	fileobj.write("Dummy\n")
	fileobj.write("Dummy2\n")
	fileobj.write("Dummy3\n")
	fileobj.write("Dummy4\n")
	fileobj.write("Dummy5\n")
	fileobj.write("Dummy6\n")
	fileobj.write("End")
	fileobj.close()
	
	
	m = read(filename_model)
	#m.Params.PoolSolutions=2000000000
	m.optimize()
	print("m.solcount=*************")
	print(m.solcount)	
	m.write("1.sol")
