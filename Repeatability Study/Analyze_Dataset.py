import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
import statsmodels.stats.api as sms
import json

# Defines the location of the survey data for this experiment
file_name='data/Repeatability Survey Data.xlsx'

# Opens the dictionary of swarm data for this experiment
with open('data/Saved_Anonymized_Repeatability_Data_Final.json') as json_file:
  dictionary = (json.load(json_file))

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    i=0
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height+.01,
                '%i' %(100*round(swarm_repeatability[i],2)),
                ha='center', va='bottom',rotation=90,size=9)
        i+=1

def CI2(confidence_level,data):	#TWO SIDED
	'''
	param data: array of data to generate CI
	param return: two entry list [low, high] of CI
	'''
	alpha=(100-confidence_level)/200
	data=sorted(data)
	data=np.array(data)
	low=alpha*(len(data))
	high=(1-alpha)*(len(data)) 
	low=int(low)
	high=int(high)
	low_value=data[low-1]
	low_value=round(low_value,3)
	high_value=data[high-1]
	high_value=round(high_value,3)
	interval=[low_value,high_value]
	return interval

def bootstrap(data,num_samples,size):
	'''
	param data: array to bootstrap over
	param num_samples: number of bootstraps
	param size: size of each bootstrap (usually len(data))
	param return: list with len(num_samples), each entry in this list is another list of len(size)
	'''
	results=[]
	for i in range(num_samples):
		bootstrap_list=[]
		rands=np.random.randint(0,len(data),size=size)
		for r in rands:
			bootstrap_list.append(data[r])
		results.append((bootstrap_list))
	return results

def individual_answers(num_individuals,df):
	'''
	param num_individuals: number of people, automatically input
	param df: survey/swarm results spreadsheet
	param return: individual answers, a list of answers by person
	'''
	Individuals_Answers=[[] for x in range(25)]
	individualslist_byperson=[]
	for i in range(num_individuals):
		personsanswers=[]
		answers= list(df.iloc[:,1+i])
		answers=np.nan_to_num(answers)
		for a in range(len(answers)):
			if 'important' in answers[a]:
				answers[a]= (answers[a][0])
			else:
				answers[a]= (answers[a][:-1])
			personsanswers.append(int(answers[a]))
			Individuals_Answers[a].append(int(answers[a]))
		individualslist_byperson.append(personsanswers)
	return Individuals_Answers,individualslist_byperson

def swarm_answer(df):
	'''
	param df: takes dataframe of survey/swarm results spreadsheet
	param return: relevant variables for analysis 
	'''
	lin_interpolation=[]
	sq_interpolation=[]
	conviction=[]
	usernames=[]
	initial_pull=[]
	interpolation_final=[]
	imp_throughtime=[]
	time_pulling_per_choice=[]
	switches=[]
	time_till_switch=[]
	switched_facs=[]
	swarm_answer= list(df.loc[:,'Swarm Answer'])
	for a in range(len(swarm_answer)):
		replay_id=str(df.loc[a,'replay_ids']) #gets replay id

		#go through json file and grab data
		lin_interpolation.append(dictionary[replay_id]['Swarm Interpolation'])
		switches.append((dictionary[replay_id]['Switch Counts']))
		conviction.append( (dictionary[replay_id]['Conviction']) )
		imp_throughtime.append(dictionary[replay_id]['Swarm Impulse Over Time'])
		usernames.append(dictionary[replay_id]['Usernames']) 
		interpolation_final.append(dictionary[replay_id]['Individual Interpolation'])
		initial_pull.append(dictionary[replay_id]['Initial Choice'])	 
		time_pulling_per_choice.append((dictionary[replay_id]['Time Per Answer Choice']))
		time_till_switch.append(dictionary[replay_id]['Time To Switch'])
		switched_facs.append(dictionary[replay_id]['Factions Switched To'])
		
	return swarm_answer,conviction,lin_interpolation,usernames,initial_pull,interpolation_final,imp_throughtime,time_pulling_per_choice,switches,time_till_switch,switched_facs

#25 = number of questions
#these are lists 25 (num of questions) long, each 1 of 25 is a list of all the groups' answers to that question
All_swarmanswers=[[] for x in range(25)]	#swarm explicit answers binned by question
All_convictions=[[] for x in range(25)]		#swarm convictions by question
All_interpolations=[[] for x in range(25)]	#swarm interpolations by question
crowd_avg_byquestion=[[] for x in range(25)]#survey average by question
imp_throughtime=[[] for x in range(25)]	#swarm impulse through time, by question

## each list is 250 long, each of 250 entries is a list of individuals, or a single entry if it is a group statistic
crowd_avg_list=[]	#survey average
interpolation_list=[]	#swarm interpolations
number_people=[]	#number of people in answered survey
usernames_list=[]	#list of usernames in swarm
All_individualanswers_list=[]	#list of individual answers on survey
initialpull_list=[]	#individuals intiial pull list
interpolation_final=[] #individuals final interpolations
switches=[]	#individuals number of switches
ind_time_per_choice=[] #percent time spent on each answer choice (including initial)
timetoswtich=[] #time for individual to switch answers
switch_facs=[]	#individuals percent time spent on each non-intiial faction

def iterate_sheets(num_sheets):
	
	for s in range(1,num_sheets+1):
		sheet='Group %s'%s
		df = pd.read_excel(file_name, sheet_name=sheet)
		cols=list(df.columns)
		num_individuals=(cols.index('Swarm Answer'))-1
		number_people.append(num_individuals)
		
		indanswer=individual_answers(num_individuals,df)
		for q in range(len(indanswer[0])):
			surveyans=[]
			for i in range(len(indanswer[0][q])):
				surveyans.append(int(indanswer[0][q][i]))
			All_individualanswers_list.append(indanswer[0][q])
			crowd_avg_byquestion[q].append(np.mean(surveyans))
			crowd_avg_list.append(np.mean(surveyans))

		swarmans=swarm_answer(df)
		for i in range(len(swarmans[0])):
			#lists by question
			All_swarmanswers[i].append(swarmans[0][i])
			All_convictions[i].append(swarmans[1][i])
			All_interpolations[i].append(swarmans[2][i])
			imp_throughtime[i].append(swarmans[6][i])

		switches.extend(swarmans[8])
		ind_time_per_choice.extend(swarmans[7])
		timetoswtich.extend(swarmans[9])
		switch_facs.extend(swarmans[10])
		interpolation_list.extend(swarmans[2])
		
		#lists by groups
		usernames_list.extend(swarmans[3])
		initialpull_list.extend(swarmans[4])
		interpolation_final.extend(swarmans[5])

iterate_sheets(10) 
	
swarm_repeatability=[]
swarm_instance_repeatability=[]
crowd_instance_repeatability=[]
for q in range(len(All_swarmanswers)):
	swarmcounts=pd.value_counts(All_swarmanswers[q])
	dictionary=dict(swarmcounts)
	swarm_repeatability.append(list(swarmcounts)[0]/sum(swarmcounts))
	for s in range(len(All_swarmanswers[q])):
		ans=All_swarmanswers[q][s]
		count= dictionary[ans]
		swarm_instance_repeatability.append((count-1)/(sum(swarmcounts)-1))

### repeatability by question graph ###
x_labels=np.arange(1,26,1)
fig, ax = plt.subplots()
rects = ax.bar(np.arange(1,26,1), swarm_repeatability,alpha=0.4)
ax.set_title('Repeatability by Question',size=16)
ax.set_xlabel('Question',size=14)
ax.set_ylabel('Repeatability',size=14)
plt.xticks(x_labels,x_labels)
autolabel(rects)
plt.show()

########### bootstrapped confidence intervals repeatability #########
questionnumbers=np.arange(0,25,1)
bootstrapresults=bootstrap(questionnumbers,5000,25)

## Repeatability ##
swarm_repeat_1000=[]
for i in bootstrapresults:
	repeatability=[]
	for x in i:
		repeatability.append(swarm_repeatability[x])
	swarm_repeat_1000.append(np.mean(repeatability))

CI_swarm_repeat=CI2(95,swarm_repeat_1000)

sns.distplot(swarm_repeat_1000,label='Swarm Mean =%.2f'%np.mean(swarm_repeat_1000))
plt.title('Swarm Repeatability \n 95%% Confidence Interval = %s'%CI_swarm_repeat)
plt.xlabel('Repeatability')
plt.ylabel('Bootstrapped Frequency')
plt.legend()
plt.show()

print('Swarm Repeatability =%s'%CI_swarm_repeat)
print('Swarm Mean Repeatability = %.3f'%np.mean(swarm_repeat_1000))

### NOT bootstrapped; average variance over 25 questions ###
question_variance=[]
for q in range(25):
	question_variance.append(np.var(All_interpolations[q]))
print('Average Intra-question Variance = %.3f'%np.mean(question_variance))

### explicit answer variance ###
swarm_ans_1000=[]
for i in bootstrapresults:
	variance_swarm_ans=[]
	for x in i:
		#appending variance of each question; bootstrapped question indices = x
		variance_swarm_ans.append(np.var(All_swarmanswers[x])) 
	#appending mean variance over those bootstrapped questions
	swarm_ans_1000.append(np.mean(variance_swarm_ans))

CI_swarm_ans=CI2(95,swarm_ans_1000)

print('Swarm Explicit Answer Variance CI = %s' %CI_swarm_ans)
print('Swarm Explicit Answer Average Variance = %.3f'%np.mean(swarm_ans_1000))

##### bootstrapped confidence intervals avg interpolation variance#####
swarm_inter_1000=[]
for i in bootstrapresults:
	variance_swarm=[]
	for x in i:
		#appending variance of each question; bootstrapped question indices = x
		variance_swarm.append(np.var(All_interpolations[x])) # divide by 4 to convert to percent binrange (1-5 range)
	#appending mean variance over those bootstrapped questions
	swarm_inter_1000.append(np.mean(variance_swarm))

CI_swarm=CI2(95,swarm_inter_1000)
print('Swarm Interpolation Variance CI = %s' %CI_swarm)
print('Swarm Interpolation Average Variance = %.3f'%np.mean(swarm_inter_1000))


## Case study example interpolation variance with grey shaded CI ###
question_index=5 # what question to use

mean = np.mean(All_interpolations[question_index])	#mean over 10 groups on that question
std=np.std(All_interpolations[question_index])	#std over 10 groups on that question

mean=round(mean,2)
x = np.arange(1,5.01,0.01)
y = stats.norm.pdf(x,loc=mean,scale=std)
conf_1 = np.logical_and(x>(mean-std),x<(mean+std))
conf_2 = np.logical_and(np.logical_and(x>(mean-2*std),x<(mean+2*std)),1-conf_1)
plt.plot(x,y)
plt.fill_between(x,y,color='grey',alpha=0.2,where=conf_2, label='2 Standard Deviations')
plt.fill_between(x,y,color='grey',alpha=0.4, where=conf_1, label='1 Standard Deviation')
plt.axvline(mean, label='Swarm Interpolation: ' + str(mean), c='k')
plt.legend()
plt.xlabel("Swarm Interpolated Response")
plt.ylabel("Probability Density")
plt.title("Confidence Interval of Interpolation for Question 6")
plt.ylim(bottom=0)
plt.show()

########## REPEATABILITY-CONVICTION GRAPH ##############
All_convictions_plot=[]
for i in All_convictions:
	All_convictions_plot.extend(i)	#making a single long list

orderedlist=sorted(zip(All_convictions_plot,swarm_instance_repeatability))	# sorting by conviction (x-axis)

ordered_repeatability=[x for _,x in orderedlist]
ordered_convictions=[x for x,_ in orderedlist]

runningavg=[]
for i in range(len(All_convictions_plot)):
	runningavg.append(np.mean(ordered_repeatability[i:i+20]))	# vary smoothness/roughnes with number of points averaged over

yhat = savgol_filter(runningavg, 15, 1)	# fit curve
plt.figure(figsize=(9.5,6))
plt.plot(sorted(All_convictions_plot),yhat)

points=[]
for i in zip(sorted(All_convictions_plot),yhat):
	if i[1]>=.9:  #count num points where fit is above 90% repeatable
		points.append(i[0])

print('Fraction points > 90%% repeatable = %s '%(len(points)/len(All_convictions_plot)))
points_percent=100*(round(points[0],2))
plt.scatter(All_convictions_plot,swarm_instance_repeatability,color='blue',label='90%% repeatable above %s%% conviction'%points_percent)
reg=stats.linregress(swarm_instance_repeatability,All_convictions_plot)
rvalue=reg[2] **2
pval = reg[3]
plt.title('Swarm Repeatability - Conviction Comparison \n R-squared = %.2f, p = %.3f'%(rvalue, pval),size=15)
plt.xlabel('Conviction',size=14)
plt.ylabel('Repeatability',size=14)
# plt.legend(loc='upper left',prop={'size': 10})
plt.grid()
plt.show()


######## Individual Behavior Analysis ########
swarm_final_interp_question=[[] for x in range(25)]	#individual interpolations, binned by question
swarm_initial_question=[[] for x in range(25)]	#individual initial pulls, binned by question
survey_initial_question=[[] for x in range(25)]	#individual answers in survey, binned by question
swarm_initial_mean=[[] for x in range(25)]	#the mean initial answer in swarm, binned by question

# each list is 250 entries (250 swarms), and each entry is a list of the individuals' data
full_swarm_initial_list=[]	#individual initial pulls (1-5 answer)
full_swarm_initial_list_index=[]	#individual pull (0-4 index)
full_survey_initial_list=[]		#individual survey answers
final_interp_list=[]		# individual final interpolations
allswitches=[]				# number of switches made by individuals in each swarm
all_time=[]					# individuals time spent pulling for each faction (0-4)
all_timetoswtich=[]			# individuals time before switching
switch_facs_list=[]			# dictionary for each individual of each non initial faction (0-4) and percent time spent on that faction
swarm_initial_mean_list=[]	#the mean initial answer in swarm
swarm_initial_median_list=[]	#the median initial answer in swarm

def iterate_individual_behavior():
	'''
	fills above lists with relevant data on individuals
	'''
	for i in range(len(initialpull_list)): #for each swarm session

		k=int(i / 25) #group number (1-10)
		question_num= (i % 25) #question index; +1 for question number, +0 for index
		df_usernames=pd.read_excel(file_name, sheet_name='Group %s'%(k+1),header=None) #excel spreadsheet with usernames
		survey_ids=list(df_usernames.iloc[0,1:])	#getting list of survey usernames in that group
		survey_answers=list(All_individualanswers_list[i])	#getting the list of their individual answers
		user_ids=usernames_list[i]	#list of usernames of people actually participating in swarm 

		## in one swarm session:
		survey_answers_swarm_users=[]
		swarm_users_initial=[]
		swarm_final_interp=[]
		ind_switches=[]
		ind_timepull=[]
		time_switch=[]
		switch_facs_percent=[]
		
		for p in range(len(user_ids)): #filters through and finds matching usernames between survey and swarm
			for s in range(len(survey_ids)): #for each person on the survey
				## getting rid of spaces and capitalization differences in data
				user_ids[p]=user_ids[p].replace(" ","")
				user_ids[p]=user_ids[p].lower()
				survey_ids[s]=survey_ids[s].replace(" ","")
				survey_ids[s]=survey_ids[s].lower()

				if user_ids[p] == survey_ids[s]: #if matches swarm username, keep their relevant data
					survey_answers_swarm_users.append(survey_answers[s])
					swarm_users_initial.append(initialpull_list[i][p])
					swarm_final_interp.append(round(interpolation_final[i][p],3))
					ind_switches.append(switches[i][p])
					ind_timepull.append(ind_time_per_choice[i][p])
					time_switch.append(timetoswtich[i][p])
					switch_facs_percent.append(switch_facs[i][p])

		## append list of individual data to master list of all 250 swarms
		survey_initial=np.array(survey_answers_swarm_users)
		swarm_initial=np.array(swarm_users_initial)
		initial_mean=np.mean(swarm_initial)			
		initial_median=int(np.median(swarm_initial))	
		swarm_initial_mean[question_num].append(initial_mean)
		swarm_initial_mean_list.append(initial_mean)
		swarm_initial_median_list.extend([initial_median]*len(switch_facs_percent))

		full_swarm_initial_list.extend(swarm_initial)
		full_swarm_initial_list_index.extend(swarm_initial -1)
		full_survey_initial_list.extend(survey_initial)
		final_interp_list.extend(swarm_final_interp)
		allswitches.extend(ind_switches)
		all_time.extend(ind_timepull)
		all_timetoswtich.extend(time_switch)
		switch_facs_list.extend(switch_facs_percent)

		swarm_initial_question[question_num].extend(swarm_initial)
		survey_initial_question[question_num].extend(survey_initial)
		swarm_final_interp_question[question_num].extend(swarm_final_interp)

iterate_individual_behavior()

def switch_median_heatmap():
	'''
	generates heatmap of where individuals switch based off initial median answer
	'''
	switch_heatmap=[[[] for x in range(9)] for y in range(9)]
	for i in range(len(switch_facs_list)):
		if len(switch_facs_list[i])>0:
			dist_from_median=full_swarm_initial_list[i]-swarm_initial_median_list[i]

			keys=list(switch_facs_list[i].keys())
			values=list(switch_facs_list[i].values())
			for k in range(len(keys)):
				switch_dist_median=int(float(keys[k])-(swarm_initial_median_list[i]-1))

				switch_heatmap[(int(switch_dist_median)+4)][int(dist_from_median)+4].append((values[k]))
	
	#### calculating net percent time on each faction ###
	percent_heatmap=[[[] for x in range(9)] for y in range(9)]
	for x in range(9):
		summedinitial=[]
		for k in range(9):
			summedinitial.extend(switch_heatmap[k][x])
		for y in range(9):
			percent=sum(switch_heatmap[y][x])/sum(summedinitial)
			percent_heatmap[y][x]=percent
	percent_heatmap_clean=np.zeros((9,9))
	for x in range(9):
		for y in range(9):
			### never possible to switch here: nan
			if (percent_heatmap[y][x])==0:
				percent_heatmap_clean[y][x]=None

			else:
				percent_heatmap_clean[y][x]=round(percent_heatmap[y][x],2)

	## happens to be no data, but in theory can be, so display 0 rather than nan
	percent_heatmap_clean[0][1]=0
	percent_heatmap_clean[8][5]=0
	percent_heatmap_clean[8][6]=0
	percent_heatmap_clean[8][7]=0

	ax=sns.heatmap(percent_heatmap_clean,annot=True,cbar_kws={'label': 'Frequency'},cmap='Blues')
	plt.title('Full Dataset: Where Individuals Switch')
	plt.xlabel('Initial Pull - Group Median')
	plt.ylabel('Support Distribution of Non-Initial Answer - Group Median')
	plt.xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5],['-4','-3','-2','-1','0','1','2','3','4'])
	plt.yticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5],['-4','-3','-2','-1','0','1','2','3','4'])
	ax.invert_yaxis()
	plt.show()	

switch_median_heatmap()

def switch_heatmap():
### heatmap of where individuals switch
	switch_heatmap2=[[[] for y in range(6)] for x in range(6)]
	for i in range(len(switch_facs_list)):
		if len(switch_facs_list[i])>0:
			
			keys=list(switch_facs_list[i].keys())
			values=list(switch_facs_list[i].values())
			for k in range(len(keys)):
				switch_heatmap2[int(float(keys[k]))][int(full_swarm_initial_list_index[i])].append((values[k]))

	#### calculating net percent time on each faction ###
	percent_heatmap2=[[[] for y in range(5)] for x in range(5)]
	for j in range(5):
		summedinitial=[]
		for k in range(len(switch_heatmap2[:][j])):
			summedinitial.extend(switch_heatmap2[k][j])
		for i in range(5):
			percent=sum(switch_heatmap2[i][j])/sum(summedinitial)
			percent_heatmap2[i][j]=percent

	percent_heatmap_clean2=np.zeros((5,5))
	for x in range(5):
		for y in range(5):
			if (percent_heatmap2[y][x])==0:
				percent_heatmap_clean2[y][x]=None
			else:
				percent_heatmap_clean2[y][x]=round(percent_heatmap2[y][x],2)
	ax=sns.heatmap(percent_heatmap_clean2,annot=True,cbar_kws={'label': 'Frequency'},cmap='Blues')
	plt.title('Full Dataset: Where Individuals Switch')
	plt.xlabel('Initial Pull')
	plt.ylabel('Support Distribution of Non-Initial Answer')
	plt.xticks([0.5,1.5,2.5,3.5,4.5],['1','2','3','4','5'])
	plt.yticks([0.5,1.5,2.5,3.5,4.5],['1','2','3','4','5'])
	ax.invert_yaxis()
	plt.show()

switch_heatmap()

def time_per_answer():
	'''
	generates graph of individuals' percent time spent on each answer distance from initial answer
	'''
	time_dict=[[] for x in range(6)] ## 6 slots: (0-4) factions & 5=nan (not pulling)
	for i in all_time:	#for each individual
		emptylist=np.zeros((6)) #time spent on each faction
		keys=list(i.keys())
		vals=list(i.values())
		for x in range(len(keys)):	
			emptylist[int(float(keys[x]))]+=(vals[x]) #percent time spent on each faction
		for f in range(len(emptylist)):
			time_dict[f].append(emptylist[f]) #putting individuals together

	final_time_dict=[]
	for i in range(len(time_dict)-1): #not using time spent on nan
		final_time_dict.append(np.mean(time_dict[i]))	# mean percent time spent on each answer over all individuals

	plt.title('Individual Support for Answers by Distance to Initial Choice')
	plt.bar(np.arange(0,5,1),final_time_dict,color='C0',alpha=.3)
	plt.xticks([0,1,2,3,4,],['0','1','2','3','4'])
	plt.xlabel('Distance from Initial Choice')
	plt.ylabel('Average Percent Time Pulling')
	plt.show()

time_per_answer()

def impulse_linear_interpolation(impulse):
	'''
	Calculates a linear interpolation for the swarm's answer given a set of impulses. The interpolation is on a 1-5 scale, with 1 as the 0th indexed target, and 5 as the 4th indexed target. This interpolation should be applied to the range in the question to make any sense.
    :param impulse: Set of impulse scores for the swarm.
    :return: linear_interpolation: Scalar value for the linear interpolation
    '''
	impulse_clean = impulse[:5] / np.sum(impulse[:5])
	return np.sum(impulse_clean*np.arange(1,6,1))

def case_study_example_question(case_question):
	'''
	param case_question: question index to use as example
	generates graphs of group interpolation over time
	prints p-values of individuals and groups
	'''
	########## Case Study  on GROUPS; Interpolation through time graph #############
	print('Case Question: Initial vs Survey by Swarms pvalue = %.6f' %(stats.ttest_rel(swarm_initial_mean[case_question],crowd_avg_byquestion[case_question])[1]))
	print('Case Question: Initial vs Final by Swarms pvalue = %.6f' %(stats.ttest_rel(swarm_initial_mean[case_question],All_interpolations[case_question])[1]))
	print('Case Question: Survey vs Final by Swarms pvalue = %.6f' %(stats.ttest_rel(All_interpolations[case_question],crowd_avg_byquestion[case_question])[1]))

	### Graph of Interpolation over time for each of 10 groups on this question ###
	initial_interpoltion=[]
	for i in range(10):
		group=i
		impulse_array=np.array((imp_throughtime[case_question][group]))
		interpolation_through_time=[]
		time=np.arange(4,len(impulse_array)+4,4) ## starting from 1 second; 4 timesteps = 1 second
		time_list=time/4 #timesteps >> seconds
		percenttime_bins=np.arange(.1,1.1,.1)

		initial_interpoltion.append(impulse_linear_interpolation(sum(impulse_array[4:12]))) ##initial interpolation from 1-3 seconds
		for t in range(len(percenttime_bins)-1):
			timestep1=int(percenttime_bins[t]*(len(impulse_array)) -1)	#lower bin
			timestep2=int(percenttime_bins[t+1]*(len(impulse_array)) -1)	#upper bin
			time_impulse=sum(impulse_array[timestep1:timestep2]) #start at one second #change lower index to 4 and upper index timestep2  if wanting cumulative interp over time
			time_interpolation=impulse_linear_interpolation(time_impulse)
			interpolation_through_time.append(time_interpolation)
		
		plt.plot(percenttime_bins[1:],interpolation_through_time,marker='o',label='Group %s'%(group+1),color='C%s'%i)
		plt.xlabel('Percent Time',size=14)
		plt.ylabel('Interpolation',size=14)

	plt.title('Case Study: Interpolation Through Time',size=16)
	plt.show()


	######## CASE STUDY on INDIVIDUALS: #######
	surveyavg=np.mean(crowd_avg_byquestion[case_question])		# survey average on this question (10 groups)
	swarmavg=np.mean(All_interpolations[case_question])			#swarm interpolation on this question (10 groups)
	swarm_init=np.array(swarm_initial_question[case_question]) 	#initial individuals' answers on this question
	survey_init=np.array(survey_initial_question[case_question])	#survey individuals' answers on this question
	final_interp=np.array(swarm_final_interp_question[case_question])	#final individuals' interpolations on this question

	print('Case Question: Initial vs Survey by Individuals pvalue = %.6f' %(stats.ttest_rel(swarm_init,survey_init)[1]))
	print('Case Question: Initial vs Final by Individuals pvalue = %.6f' %(stats.ttest_rel(swarm_init,final_interp)[1]))
	print('Case Question: Survey vs Final by Individuals pvalue = %.6f' %(stats.ttest_rel(survey_init,final_interp)[1]))

	### making dictionary of faction support frequency ###
	countsvals=(dict(pd.value_counts(survey_init)))
	countsvals2=dict(pd.value_counts(swarm_init))
	values1=[]
	for i in list(countsvals.keys()):
		values1.append(i+.15) #offsetting bins to graph side by side
	values2=[]
	for i in list(countsvals2.keys()):
		values2.append(i-.15) #offsetting bins to graph side by side
	frac_vals=[]
	for i in range(len(countsvals.values())):
		frac_vals.append( list(countsvals.values())[i]/(sum(countsvals.values())))
	frac_vals_2=[]
	for i in range(len(countsvals2.values())):
		frac_vals_2.append(list(countsvals2.values())[i]/(sum(countsvals2.values())))

	#### Graph of Individuals' Survey, Swarm Initial, and Swarm Final Interpolation ### 
	plt.title('Case Question: Individuals Answers')
	plt.bar(values1,frac_vals,width=.3,color='C0',alpha=.4,label='Survey: Mean = %.1f, std = %.2f'%(np.mean(survey_init),np.std(survey_init)))
	plt.bar(values2,frac_vals_2,width=.3,color='C1',alpha=.4,label='Swarm Initial: Mean = %.1f, std = %.2f'%(np.mean(swarm_init),np.std(swarm_init)))
	weights = np.ones_like(final_interp)/float(len(final_interp))
	plt.hist(final_interp,label='Swarm Final: Mean = %.1f, std = %.2f'%(np.mean(final_interp),np.std(final_interp)),color='C2',bins=np.arange(1,5,.3),alpha=.4,weights=weights)
	plt.xlabel('Answer')
	plt.ylabel('Frequency')
	plt.legend()
	plt.show()

	### individual mean (mean diff, CI) of swarm initial vs swarm final ###
	diff=np.array(final_interp) - np.array(swarm_init)	
	print('Case Question: Mean Final - Initial Mean Difference = %.3f , p = %.3f'%(np.mean(diff), stats.ttest_rel(final_interp,swarm_init)[1]))

	#### bootstrapped individual standard deviation (mean diff, CI) of swarm initial vs swarm final ###
	bootstrapped=bootstrap(range(len(swarm_init)),1000,len(swarm_init)) ## bootstrapping individuals
	init_std=[]
	final_std=[]
	for i in range(len(bootstrapped)):
		init_std .append(np.std(list(swarm_init[bootstrapped[i]])) ) #std of individuals initially
		final_std.append(np.std(list(final_interp[bootstrapped[i]])) ) #std of individuals final
	p_val=stats.ttest_rel(init_std,final_std)[1]
	print('Case Question: Mean Final - Initial Standard Deviation Difference = %.3f, p= %.3f'%(np.mean(np.array(final_std)-np.array(init_std)),p_val))

# Creates a graph of interpolation through time for Question 23
case_study_example_question(23)

# Creates a graph of interpolation through time for Question 23
case_study_example_question(4)

############## p value comparisons full test ####################
## by groups (10 datapoints) ##
initial_group_mean=swarm_initial_mean_list
final_group_interpolation=interpolation_list
survey_groups=crowd_avg_list
print('FULL DATASET: Initial vs Survey by Swarms pvalue = %.6f, diff = %.3f' %(stats.ttest_rel(initial_group_mean,survey_groups)[1], np.mean(np.array(initial_group_mean)-survey_groups)))
print('FULL DATASET: Initial vs Final by Swarms pvalue = %.6f, diff = %.3f' %(stats.ttest_rel(initial_group_mean,final_group_interpolation)[1], np.mean(np.array(initial_group_mean)-final_group_interpolation)))
print('FULL DATASET: Survey vs Final by Swarms pvalue = %.6f, diff = %.3f' %(stats.ttest_rel(survey_groups,final_group_interpolation)[1], np.mean(np.array(survey_groups)-final_group_interpolation)))

## by individuals (4033 datapoints) ##
print('FULL DATASET: Initial vs Survey by Individuals pvalue = %.6f, diff = %.3f' %(stats.ttest_rel(full_swarm_initial_list,full_survey_initial_list)[1], np.mean(np.array(full_swarm_initial_list)-full_survey_initial_list)))
print('FULL DATASET: Initial vs Final by Individuals pvalue = %.6f, diff = %.3f' %(stats.ttest_rel(full_swarm_initial_list,final_interp_list)[1], np.mean(np.array(full_swarm_initial_list)-final_interp_list)))
print('FULL DATASET: Survey vs Final by Individuals pvalue = %.6f, diff = %.3f' %(stats.ttest_rel(full_survey_initial_list,final_interp_list)[1], np.mean(np.array(full_survey_initial_list)-final_interp_list)))

