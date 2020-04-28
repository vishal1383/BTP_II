from utils import *
import numpy as np
import math
from sklearn.decomposition import PCA
from projpursuit import projpursuit
from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon

def var (direction,axes):
    direction = np.asarray(direction)
    direction = direction/np.linalg.norm(direction)
    for i in range(len(axes)):
        axes[i] = axes[i] - np.dot(axes[i],direction)*direction
    return np.linalg.norm(np.std(axes, axis=0))

def top_k (axes,k):
    PCs = []
    pca = PCA(n_components=len(axes))
    principalComponents = pca.fit_transform(axes)
    for i in range(len(axes)):
        PCs.append([pca.components_[i],var(pca.components_[i],axes)])
    PCs = sorted(PCs, key = lambda x: x[1],reverse=True)
    for i in range(len(PCs)):
        #print (PCs[i][1]) #    variance value changing everytime, why?
        PCs[i]= PCs[i][0]
    return (PCs[:k])

def top_k_PPA (axes,k):
    X = np.random.rand(len(axes), len(axes[0]))
    X/=1000000000
    for i in range(len(axes)):
        for j in range(len(axes[i])):
            X[i][j]+=axes[i][j]
    X=X.T
    T, V, PPOUT = projpursuit(X, p=len(axes)-1)
    T=T.T
    PPs = []
    for i in range(len(axes)-1):
        PPs.append([T[i],var(T[i],axes)])
    PPs = sorted(PPs, key = lambda x: x[1],reverse=True)
    for i in range(len(PPs)):
        PPs[i]= PPs[i][0]
    return (PPs[:k])
	
def debias(g, N, candidates):
    dic = {}
    for i in N:
        word = np.asarray(emb_full[i])/(np.linalg.norm(np.asarray(emb_full[i])))
        for g_i in g:
            g_i = g_i/np.linalg.norm(g_i)
            word = word - np.dot(word,g_i)*g_i
        dic[i]=word

    for (a,b) in candidates:
        dic[a] = emb_full[a]
        dic[b] = emb_full[b]
        vec=(dic[a] + dic[b]) / 2
        for g_i in g:
            g_i = g_i/np.linalg.norm(g_i)
            vec = vec - np.dot(word,g_i)*g_i
            
        z = np.sqrt(1 - np.linalg.norm(vec)**2)
        component,diff=0,dic[a]-dic[b]
        lis=[]
        for g_i in g:
            g_i = g_i/np.linalg.norm(g_i)
            component += np.dot(diff,g_i)
            lis.append(abs(np.dot(diff,g_i)))

        lis=[(i/sum(lis))**0.5 for i in lis]
        if component < 0:
            z = -z
        dic[a],dic[b]=vec,vec
        j_idx=0
        for g_i in g:
            dic[a] += lis[j_idx]* z * g_i
            dic[b] += -1*lis[j_idx]* z * g_i
            j_idx+=1

    return dic
   
def cosine1(x, y):
    return np.dot(x,y.T)/(np.linalg.norm(x)*np.linalg.norm(y))

def return_test_sets():
    Adj_en = ['fair', 'good', 'light', 'nervous', 'rich', 'modern', 'plain', 'important', 'special', 'powerful', 'bright', 'evil', 'alert', 'concerned', 'unusual', 'odd', 'new', 'little', 'low', 'big', 'high', 'difficult', 'easy', 'kind', 'famous', 'friendly', 'helpful', 'careful', 'nice', 'perfect', 'better', 'young', 'happy', 'dark', 'average', 'super', 'hard', 'small', 'wild', 'poor', 'confused', 'fat', 'thin', 'long', 'clean', 'strong', 'large', 'bad', 'strange', 'old']
    Adj_hi = ['गंभीर', 'महत्वाकांक्षी', 'आक्रामक', 'यथार्थवादी', 'उदारवादी', 'आशावादी', 'संकीर्ण', 'कम', 'आकृतियों', 'परिपत्र', 'वर्ग', 'त्रिकोणीय', 'स्वाद', 'नमकीन', 'मसालेदार', 'कठिन', 'स्वच्छ', 'आसान', 'उपवास', 'भारी', 'प्रकाश', 'स्थानीय', 'शक्तिशाली', 'शांत', 'सही', 'बहुत', 'कमज़ोर', 'मात्राएँ', 'अनेक', 'नारंगी', 'लाल', 'सफ़ेद', 'आकार', 'अनुशासित', 'सरल', 'दयालु', 'ईमानदार', 'मेहनती', 'आराध्य', 'स्मार्ट', 'परिष्कृत', 'पराक्रमी', 'प्रेरित', 'व्यापक']
    Adj_be = ['গম্ভীর', 'উচ্চাকাঙ্ক্ষী', 'আক্রমনাত্মক', 'বাস্তবানুগ', 'মধ্যপন্থী', 'আশাবাদী', 'কম', 'সঙ্কীর্ণ', 'আকার', 'বিজ্ঞপ্তি', 'বর্গাকার', 'স্বাদ', 'নোনতা', 'মশলাদার', 'কঠিন', 'পরিষ্কার', 'সহজ', 'উপোস', 'ভারী', 'প্রকাশ', 'স্থানীয়', 'শক্তিশালী', 'শান্ত', 'খুব', 'দুর্বল', 'অনেক', 'কমলা', 'লাল', 'সাদা', 'আকৃতি', 'আকার', 'নিয়মনিষ্ঠ', 'সহজ', 'সদয়', 'পরিশ্রমী', 'আরাধ্য', 'স্মার্ট', 'বাস্তববুদ্ধিসম্পন্ন', 'মহৎ', 'অনুপ্রাণিত', 'ব্যাপক', 'স্বচ্ছ', 'সত্যি', 'পরাক্রমী', 'পরিষ্কৃত']
    Adj_te = ['సీరియస్', 'ప్రతిష్టాత్మక', 'దూకుడు', 'యదార్థ', 'నియంత్రించు', 'ఆప్టిమిస్టిక్', 'నిశితం', 'తక్కువ', 'ఆకారాలు', 'సర్క్యులర్', 'చదరపు', 'రుచి', 'ఉప్పగా', 'ఉండే', 'తెలంగాణ', 'కష్టం', 'క్లీన్', 'సులువు', 'ఉపవాసం', 'భారీ', 'కాంతి', 'స్థానిక', 'శక్తివంతమైన', 'శాంతిగా', 'హక్కు', 'చాలా', 'బలహీనుడు', 'వాల్యూమ్స్', 'అనేక', 'నారింజ', 'ఎరుపు', 'వైట్', 'ఆకారం', 'క్రమశిక్షణ', 'సాధారణ', 'నిజాయితీ', 'కష్టపడి', 'పూజ్యమైన', 'స్మార్ట్', 'అధునాతన', 'ప్రేరణ', 'సమగ్ర', 'లిగా', 'వేడిగా', 'మబ్బుగా', 'వర్షంగా', 'మంచుగా', 'ఎండగా', 'గాలిగా', 'వేడిగా', 'శక్తివంతమైన', 'శాంతమైన', 'సరైన', 'మెల్లని', 'మెత్తని', 'ఎక్కువ', 'బలహీనమైన', 'తప్పు', 'చిన్నవయసు', 'పరిమాణములు', 'కొద్ది', 'చాలా', 'భాగం', 'కొన్ని', 'కొంచెం', 'మొత్తం']

    NE = ['professor', 'administrator', 'ambassador', 'architect', 'artist', 'dean', 'athlete', 'director', 'author', 'baker', 'boss', 'driver', 'captain', 'chancellor', 'clerk', 'coach', 'collector', 'commander', 'commissioner', 'composer', 'conductor', 'consultant', 'cop', 'critic', 'deputy', 'doctor', 'student', 'editor', 'farmer', 'pilot', 'writer', 'designer', 'guitarist', 'historian', 'journalist', 'judge', 'lieutenant', 'manager', 'mathematician', 'minister', 'musician', 'novelist', 'officer', 'philosopher', 'poet', 'politician', 'president', 'principal', 'saint', 'scholar', 'scientist', 'secretary', 'singer', 'songwriter', 'soldier', 'steward', 'substitute', 'teacher', 'warrior']
    NH = ['वास्तुकार', 'राजगीर', 'दर्जी', 'माली', 'पत्रकार', 'न्यायाधीश', 'वकील', 'व्यवस्थापक', 'राजदूत', 'कलाकार', 'आयुक्त', 'संगीतकार', 'सलाहकार', 'आलोचक', 'उप', 'छात्र', 'संपादक', 'इतिहासकार', 'प्रबंधक', 'गणितज्ञ', 'मंत्री', 'उपन्यासकार', 'अधिकारी', 'दार्शनिक', 'राजनीतिज्ञ', 'अध्यक्ष', 'संत', 'सचिव']
    NB = ['রাজমিস্ত্রি', 'দর্জি', 'মালী', 'সাংবাদিক', 'বিচারক', 'উকিল', 'অ্যাডমিন', 'রাষ্ট্রদূত', 'শিল্পী', 'কমিশনার', 'সুরকার', 'পরামর্শকারী', 'সমালোচক', 'উপাচার্য', 'শিক্ষার্থী', 'সম্পাদক', 'ইতিহাসবেত্তা', 'ম্যানেজার', 'গণিতবিদ', 'মন্ত্রী', 'ঔপন্যাসিক', 'অফিসার', 'দার্শনিক', 'রাজনীতিজ্ঞ', 'চেয়ারম্যান', 'সাধু', 'সচিব', 'নির্দেশক', 'গীতিকার']    
    NT = ['న్యాయమూర్తి', 'న్యాయవాది', 'నిర్వాహకుడు', 'రాయబారి', 'కళాకారుడు', 'కమిషనర్', 'సంగీతకారుడు', 'కన్సల్టెంట్', 'విమర్శకుడు', 'సబ్', 'విద్యార్థులు', 'ఎడిటర్', 'చరిత్రకారుడు', 'మేనేజర్', 'గణిత','శాస్త్రవేత్తలు', 'మంత్రి', 'నవలా', 'రచయిత', 'ఆఫీసర్', 'ఫిలాసఫర్స్', 'రాజకీయ', 'చైర్మన్', 'సెయింట్', 'కార్యదర్శి', 'విద్యార్థి', 'పాఠశాల']

    A = ['जज', 'लेफ्टिनेंट', 'प्रिंसिपल', 'कंडक्टर', 'नर्स', 'प्रोफेसर', 'पायलट', 'बेकर', 'ड्राइवर', 'कप्तान', 'चांसलर', 'क्लर्क', 'कलेक्टर', 'कमांडर', 'डेल्टा', 'एनीमे', 'स्पा', 'जेलीफ़िश', 'रजिस्ट्री', 'अलास्का', 'बायोमेडिकल', 'अल्ट्रासाउंड', 'एयरोस्पेस', 'क्रिकेट', 'डेल्टा']
    C = ['বিচারক', 'লেফটানেন্ট', 'অধ্যক্ষ', 'প্রিন্সিপাল', 'কন্ডাকটর', 'নার্স', 'অধ্যাপক', 'পাইলট', 'চালক', 'অধিনায়ক', 'আচার্য', 'কেরানী', 'সংগ্রাহক', 'সেনাপতি', 'সেনাধ্যক্ষ']

    E = NE + Adj_en
    H = NH + Adj_hi + A
    B = NB + Adj_be + C
    T = NT + Adj_te

    All = E + H + B + T
    # print("lengths: \nAdj: ", len(Adj_en), len(Adj_hi), len(Adj_be), 
    #     '\nN: ', len(NE), len(NH), len(NB), 
    #     '\nothers: ', 0, len(A), len(B),
    #     '\nFull lang: ', len(E), len(H), len(B),
    #     '\nAll: ', len(All))
    return E, H, B, T, All

def return_gender_pairs():
    G_hi = [['मर्द', 'औरत'], ['लड़का', 'लड़की'], ['आदमी', 'महिला'], ['पुरुष', 'स्त्री'], ['बेटा', 'बेटी'], ['पति', 'पत्नी'], ['मां', 'पिता'], ['चाचा', 'चाची'], ['मामा', 'मामी'], ['राजा', 'रानी'], ['नाना', 'नानी'], ['भाई', 'बहन'], ['भैया', 'भाभी'], ['देवर', 'देवरानी'], ['देव', 'देवी'], ['पुत्र', 'पुत्री'], ['श्रीमान', 'श्रीमति'], ['बालक', 'बालिका'], ['भतीजा', 'भतीजी'], ['नाना', 'नानी'], ['साधु', 'साध्वी']]
    G_be = [['পুরুষ', 'মহিলা'], ['ছেলে', 'মেয়ে'], ['পুত্র', 'কন্যা'], ['স্বামী', 'স্ত্রী'], ['মা', 'বাবা'], ['মামা', 'মামী'], ['রাজা', 'রানী'], ['দাদু', 'দিদা'], ['ভাই', 'বোন'], ['দাদা', 'বৌদি'], ['দেব', 'দেবী'], ['পুত্র', 'কন্যা'], ['শ্রীমান', 'শ্রীমতি'], ['ভাইপো', 'ভাইঝি'], ['বোনপো', 'বোনঝি'], ['ঠাকুরদা', 'ঠাকুমা'], ['সন্ন্যাসী', 'সন্ন্যাসিনী'], ['কাকা', 'কাকী'], ['দেওর', 'জা'], ['নন্দাই', 'ননদ'], ['শালা', 'শালী']]
    G_en = [['he', 'she'], ['boy', 'girl'], ['boys', 'girls'], ['brother', 'sister'], ['brothers', 'sisters'], ['king', 'queen'], ['kings', 'queens'], ['male', 'female'], ['man', 'woman'], ['men', 'women'], ['son', 'daughter'], ['his', 'her'], ['prince', 'princess'], ['gal', 'guy'], ['actor', 'actress'], ['husband', 'wife'], ['father', 'mother'], ['god', 'goddess']]
    G_te = [['మగ', 'ఆడ'], ['కొడుకు', 'కుమార్తె'], ['భర్త', 'భార్య'], ['తల్లి', 'తండ్రి'], ['అంకుల్', 'అత్త'], ['రాజు', 'రాణి'], ['సోదరుడు', 'సోదరి'], ['సోదరుడు', 'బావ'], ['బ్రదర్', 'సోదరి'], ['దేవుడు', 'దేవత'], ['కొడుకు', 'కుమార్తె'], ['మిస్టర్', 'శ్రీమతి'], ['అబ్బాయి', 'అమ్మాయి'], ['మేనల్లుడు', 'మేనకోడలు'], ['ఎద్దు', 'ఆవు']]
    
    hi_pairs_train = [['मर्द', 'औरत'], ['लड़का', 'लड़की'], ['आदमी', 'महिला'], ['पुरुष', 'स्त्री'], ['बेटा', 'बेटी'], ['पति', 'पत्नी'], ['मां', 'पिता'], ['चाचा', 'चाची'], ['मामा', 'मामी'], ['राजा', 'रानी']]
    hi_pairs_test = [['भाई', 'बहन'], ['भैया', 'भाभी'], ['देवर', 'देवरानी'], ['देव', 'देवी'], ['पुत्र', 'पुत्री'], ['श्रीमान', 'श्रीमति'], ['बालक', 'बालिका'], ['भतीजा', 'भतीजी'], ['नाना', 'नानी'], ['साधु', 'साध्वी']]    
    
    en_pairs_train = [['he', 'she'], ['boy', 'girl'], ['brother', 'sister'], ['king', 'queen'], ['male', 'female'], ['man', 'woman'], ['son', 'daughter'], ['his', 'her'], ['prince', 'princess'], ['actor', 'actress']]
    en_pairs_test = [['boys', 'girls'], ['brothers', 'sisters'], ['kings', 'queens'], ['men', 'women'], ['gal', 'guy'], ['husband', 'wife'], ['father', 'mother'], ['god', 'goddess'], ['father-in-law', 'mother-in-law'], ['son-in-law', 'daughter-in-law']]

    be_pairs_train = [['পুরুষ', 'মহিলা'], ['ছেলে', 'মেয়ে'], ['পুত্র', 'কন্যা'], ['স্বামী', 'স্ত্রী'], ['মা', 'বাবা'], ['মামা', 'মামী'], ['রাজা', 'রানী'], ['দাদু', 'দিদা'], ['ভাই', 'বোন'], ['দাদা', 'বৌদি']]
    be_pairs_test = [['দেব', 'দেবী'], ['পুত্র', 'কন্যা'], ['শ্রীমান', 'শ্রীমতি'], ['ভাইপো', 'ভাইঝি'], ['বোনপো', 'বোনঝি'], ['ঠাকুরদা', 'ঠাকুমা'], ['সন্ন্যাসী', 'সন্ন্যাসিনী'], ['কাকা', 'কাকী'], ['দেওর', 'জা'], ['নন্দাই', 'ননদ'], ['শালা', 'শালী']]

    te_pairs_train = [['మగ', 'ఆడ'], ['కొడుకు', 'కుమార్తె'], ['భర్త', 'భార్య'], ['తల్లి', 'తండ్రి'], ['అంకుల్', 'అత్త'], ['రాజు', 'రాణి'], ['సోదరుడు', 'సోదరి'], ['సోదరుడు', 'బావ'], ['బ్రదర్', 'సోదరి'], ['దేవుడు', 'దేవత']]
    te_pairs_test = [['కొడుకు', 'కుమార్తె'], ['మిస్టర్', 'శ్రీమతి'], ['అబ్బాయి', 'అమ్మాయి'], ['మేనల్లుడు', 'మేనకోడలు'], ['ఎద్దు', 'ఆవు']]

    axes_hi = []
    axes_en = []
    axes_be = []
    axes_te = []

    for i in hi_pairs_train:
        temp = np.asarray(emb_full[i[1]])- np.asarray(emb_full[i[0]])
        axes_hi.append(temp/np.linalg.norm(temp))
    for i in en_pairs_train:
        temp = np.asarray(emb_full[i[1]])- np.asarray(emb_full[i[0]])
        axes_en.append(temp/np.linalg.norm(temp))
    for i in be_pairs_train:
        temp = np.asarray(emb_full[i[1]])- np.asarray(emb_full[i[0]])
        axes_be.append(temp/np.linalg.norm(temp))
    for i in te_pairs_train:
        temp = np.asarray(emb_full[i[1]])- np.asarray(emb_full[i[0]])
        axes_te.append(temp/np.linalg.norm(temp))

    axes_all = []
    axes_all.extend(axes_hi)
    axes_all.extend(axes_en)
    axes_all.extend(axes_be)
    axes_all.extend(axes_te)
    
    e_plusE,e_minusE = [],[]
    for i in en_pairs_test:
        e_minusE.append(i[0])
        e_plusE.append(i[1])
    e_plusH,e_minusH = [],[]
    for i in hi_pairs_test:
        e_minusH.append(i[1])
        e_plusH.append(i[0])
    e_plusB,e_minusB = [],[]
    for i in be_pairs_test:
        e_minusB.append(i[1])
        e_plusB.append(i[0])        
    e_plusT,e_minusT = [],[]
    for i in te_pairs_test:
        e_minusT.append(i[1])
        e_plusT.append(i[0])        
    
    e_plus = e_plusH + e_plusE + e_plusB + e_plusT
    e_minus = e_minusH + e_minusE + e_minusB + e_minusT
    
    #print("English train pairs: ", en_pairs_train, len(en_pairs_train), "\nEnglish test pairs: ", en_pairs_test, len(en_pairs_test))
    #print("Hindi train pairs: ", hi_pairs_train, len(hi_pairs_train), "\nHindi test pairs: ", hi_pairs_test, len(hi_pairs_test))
    #print("Bengali train pairs: ", be_pairs_train, len(be_pairs_train), "\nBengali test pairs: ", be_pairs_test, len(be_pairs_test))
    #print("Telegu train pairs: ", be_pairs_train, len(be_pairs_train), "\nBengali test pairs: ", be_pairs_test, len(be_pairs_test))
    return axes_en, axes_hi, axes_be, axes_te, axes_all, e_plusH, e_minusH, e_plusE, e_minusE, e_plusB, e_minusB, e_plusT, e_minusT, e_plus, e_minus, G_hi,G_be,G_en,G_te


def ECT(emb,E_plus,E_minus,neutral_list_words):
    vec_plus,vec_minus=np.array([0]*300,dtype=np.float),np.array([0]*300,dtype=np.float)
    for e in E_plus:
        vec_plus+=emb[e]     
    for e in E_minus:
        vec_minus+=emb[e]
    vec_plus /= len(E_plus)
    vec_minus /= len(E_minus)
    sim1= []
    sim2=[]
    for word in neutral_list_words:
        try:
            sim1.append(1-cosine1(vec_plus,emb[word]))
            sim2.append(1-cosine1(vec_minus,emb[word]))
        except:
            pass
    return spearmanr(sim1, sim2)

def new2(emb,E_plus,E_minus,neutral_list_words, epsilon):
    def overlap(alpha, beta, epsilon):
        if(abs(alpha-beta)<2*epsilon):
            t = min(alpha, beta)
            alpha = max(alpha, beta)
            beta = t
            return ((beta+epsilon)-(alpha-epsilon))/(2*epsilon)
        return 0
    E_plus=[i for i in E_plus if i in emb.keys()]
    E_minus=[i for i in E_minus if i in emb.keys()]
    neutral_list_words=[i for i in neutral_list_words if i in emb.keys()]
    vec_plus,vec_minus=np.array([0]*300,dtype=np.float),np.array([0]*300,dtype=np.float)
    for e in E_plus:
        vec_plus+=emb[e]     
    for e in E_minus:
        vec_minus+=emb[e]
    vec_plus /= len(E_plus)
    vec_minus /= len(E_minus)
    sim = []
    for word in neutral_list_words:
        alpha = 1-cosine1(vec_plus,emb[word])
        beta = 1-cosine1(vec_minus,emb[word])
        #max_val = abs(alpha-beta)
        sim.append(overlap(alpha, beta, epsilon))
    return sum(sim)/len(sim)


def new3(emb, E_plus, E_minus, neutral_list_words):
    vec_plus,vec_minus=np.array([0]*300,dtype=np.float),np.array([0]*300,dtype=np.float)
    for e in E_plus:
        vec_plus+=emb[e]     
    for e in E_minus:
        vec_minus+=emb[e]
    vec_plus /= len(E_plus)
    vec_minus /= len(E_minus)

    male_sim,female_sim=[],[]
    for word in neutral_list_words:
        alpha = 1-cosine1(vec_plus,emb[word])
        beta = 1-cosine1(vec_minus,emb[word])
        male_sim.append(alpha)
        female_sim.append(beta)
    return jensenshannon(male_sim,female_sim)

def run_tests(embeddings_list, K_GENDER):
    
    NE, NH, NB, NT, N = return_test_sets()
    axis_en, axis_hi, axis_be, axis_te, axis_all, e_plusH, e_minusH, e_plusE, e_minusE, e_plusB, e_minusB, e_plusT, e_minusT, e_plus, e_minus,G_hi,G_be,G_en,G_te = return_gender_pairs()
    G_all=[]
    G_all.extend(G_hi)
    G_all.extend(G_be)
    G_all.extend(G_en)
    G_all.extend(G_te)
    hi_HD = top_k(axis_hi,k=1)
    en_HD = top_k(axis_en,k=1)
    be_HD = top_k(axis_be,k=1)
    te_HD = top_k(axis_te, k=1)
    all_HD = top_k(axis_all, k=1)

    hi_HD_PPA = [np.array(top_k_PPA(axis_hi,k=2)[0])]
    en_HD_PPA = [np.array(top_k_PPA(axis_en,k=2)[0])]
    be_HD_PPA = [np.array(top_k_PPA(axis_be,k=2)[0])]
    te_HD_PPA = [np.array(top_k_PPA(axis_te,k=2)[0])]
    all_HD_PPA = [np.array(top_k_PPA(axis_all,k=2)[0])]

    embed_hi_HD = debias(hi_HD, N,G_all)
    embed_en_HD = debias(en_HD, N,G_all)
    embed_be_HD = debias(be_HD, N,G_all)
    embed_te_HD = debias(te_HD, N,G_all)
    embed_all_HD = debias(all_HD, N,G_all)

    embed_hi_HD_PPA = debias(hi_HD_PPA, N,G_all)
    embed_en_HD_PPA = debias(en_HD_PPA, N,G_all)
    embed_be_HD_PPA = debias(be_HD_PPA, N,G_all)
    embed_te_HD_PPA = debias(te_HD_PPA, N,G_all)
    embed_all_HD_PPA = debias(all_HD_PPA, N,G_all)

    g_hi_PCA = top_k(axis_hi,k=K_GENDER)
    g_en_PCA = top_k(axis_en,k=K_GENDER)
    g_be_PCA = top_k(axis_be,k=K_GENDER)
    g_te_PCA = top_k(axis_te,k=K_GENDER)
    g_all_PCA = top_k(axis_all, k=K_GENDER)
    g_all_PCA_temp = top_k(axis_all, K_GENDER*3)    
    g_eq_rep = []
    temp = []
    for i in g_all_PCA_temp:
        sum_hi = 0
        sum_en = 0
        sum_be = 0
        sum_te = 0
        for j in g_hi_PCA:
            sum_hi = sum_hi + cosine1(i,j)
        sum_hi = sum_hi / len(g_hi_PCA)
        for j in g_en_PCA:
            sum_en = sum_en + cosine1(i,j)
        sum_en = sum_en / len(g_en_PCA)
        for j in g_be_PCA:
            sum_be = sum_be + cosine1(i,j)
        sum_be = sum_be / len(g_be_PCA)
        for j in g_te_PCA:
            sum_te = sum_te + cosine1(i,j)
        sum_te = sum_te / len(g_te_PCA)
        if sum_en == max(sum_en, sum_hi, sum_be, sum_te):
            temp.append('en')
        elif sum_hi == max(sum_en, sum_hi, sum_be, sum_te):
            temp.append('hi')
        elif sum_be == max(sum_en, sum_hi, sum_be, sum_te):
            temp.append('be')
        else:
            temp.append('te')
    print("g_all_PCA_temp",K_GENDER,temp)
    count=int(K_GENDER/4)
    for i in range(len(temp)):
        if temp[i]=='hi':
            g_eq_rep.append(g_all_PCA_temp[i])
            count-=1
        if count==0:
            break
    count=int(K_GENDER/4)
    for i in range(len(temp)):
        if temp[i]=='en':
            g_eq_rep.append(g_all_PCA_temp[i])
            count-=1
        if count==0:
            break
    count=int(K_GENDER/4)
    for i in range(len(temp)):
        if temp[i]=='be':
            g_eq_rep.append(g_all_PCA_temp[i])
            count-=1
        if count==0:
            break    
    count=int(K_GENDER/4)
    for i in range(len(temp)):
        if temp[i]=='te':
            g_eq_rep.append(g_all_PCA_temp[i])
            count-=1
        if count==0:
            break    

    embed_g_hi_PCA = debias(g_hi_PCA, N,G_all)
    embed_g_en_PCA = debias(g_en_PCA, N,G_all)
    embed_g_be_PCA = debias(g_be_PCA, N,G_all)
    embed_g_te_PCA = debias(g_te_PCA, N,G_all)
    embed_g_all_PCA = debias(g_all_PCA, N,G_all)
    embed_g_eq_rep_PCA = debias(g_eq_rep, N,G_all)

    g_hi_PPA = top_k_PPA(axis_hi,k=K_GENDER)
    g_en_PPA = top_k_PPA(axis_en,k=K_GENDER)
    g_be_PPA = top_k_PPA(axis_be,k=K_GENDER)
    g_te_PPA = top_k_PPA(axis_te,k=K_GENDER)
    g_all_PPA = top_k_PPA(axis_all, k=K_GENDER)
    g_all_PPA_temp = top_k_PPA(axis_all, k=2*K_GENDER)    
    g_eq_rep = []
    temp = []
    for i in g_all_PPA_temp:
        sum_hi = 0
        sum_en = 0
        sum_be = 0
        sum_te = 0
        for j in g_hi_PPA:
            sum_hi = sum_hi + cosine1(i,j)
        sum_hi = sum_hi / len(g_hi_PPA)
        for j in g_en_PPA:
            sum_en = sum_en + cosine1(i,j)
        sum_en = sum_en / len(g_en_PPA)
        for j in g_be_PPA:
            sum_be = sum_be + cosine1(i,j)
        sum_be = sum_be / len(g_be_PPA)
        for j in g_te_PPA:
            sum_te = sum_te + cosine1(i,j)
        sum_te = sum_te / len(g_te_PPA)
        if sum_en == max(sum_en, sum_hi, sum_be, sum_te):
            temp.append('en')
        elif sum_hi == max(sum_en, sum_hi, sum_be, sum_te):
            temp.append('hi')
        elif sum_be == max(sum_en, sum_hi, sum_be, sum_te):
            temp.append('be')
        else:
            temp.append('te')
    print("g_all_PPA_temp",K_GENDER,temp)
    count=int(K_GENDER/4)
    for i in range(len(temp)):
        if temp[i]=='hi':
            g_eq_rep.append(g_all_PPA_temp[i])
            count-=1
        if count==0:
            break
    count=int(K_GENDER/4)
    for i in range(len(temp)):
        if temp[i]=='en':
            g_eq_rep.append(g_all_PPA_temp[i])
            count-=1
        if count==0:
            break
    count=int(K_GENDER/4)
    for i in range(len(temp)):
        if temp[i]=='be':
            g_eq_rep.append(g_all_PPA_temp[i])
            count-=1
        if count==0:
            break
    count=int(K_GENDER/4)
    for i in range(len(temp)):
        if temp[i]=='te':
            g_eq_rep.append(g_all_PPA_temp[i])
            count-=1
        if count==0:
            break  

    embed_g_hi_PPA = debias(g_hi_PPA, N,G_all)
    embed_g_en_PPA = debias(g_en_PPA, N,G_all)
    embed_g_be_PPA = debias(g_be_PPA, N,G_all)
    embed_g_te_PPA = debias(g_te_PPA, N,G_all)
    embed_g_all_PPA = debias(g_all_PPA, N,G_all)
    embed_g_eq_rep_PPA = debias(g_eq_rep, N,G_all)

    es = {'hi':[e_plusH, e_minusH], 'en':[e_plusE, e_minusE], 'bn':[e_plusB, e_minusB], 'te':[e_plusT, e_minusT],'all':[e_plus, e_minus]}
    tests = {'hi':NH, 'en':NE, 'bn':NB, 'te':NT, 'all':N}
    embeddings = {'Orig':emb_full,
                  'hi_HD': embed_hi_HD, 'en_HD':embed_en_HD, 'bn_HD':embed_be_HD, 'te_HD':embed_te_HD, 'all_HD':embed_all_HD,
                  'hi_HD_PPA': embed_hi_HD_PPA, 'en_HD_PPA':embed_en_HD_PPA, 'bn_HD_PPA':embed_be_HD_PPA, 'te_HD_PPA':embed_te_HD_PPA, 'all_HD_PCA':embed_all_HD_PPA,
                  'g_hi_PCA':embed_g_hi_PCA, 'g_en_PCA':embed_g_en_PCA, 'g_bn_PCA':embed_g_be_PCA, 'g_te_PCA':embed_g_te_PCA,
                  'g_hi_PPA':embed_g_hi_PPA, 'g_en_PPA':embed_g_en_PPA, 'g_be_PPA':embed_g_be_PPA, 'g_te_PPA':embed_g_te_PPA,
                  'equal_rep_PCA':embed_g_eq_rep_PCA, 'equal_rep_PPA':embed_g_eq_rep_PPA,
                  'all_PCA':embed_g_all_PCA, 'all_PPA':embed_g_all_PPA,
                }                

    # ECT is actually mode-ect
    # Ignore mod-ect func
    # new2 is overlap approach
    
    K_MOD_ECT = 5
    EPSILON = 0.2
    for test in tests:
        for e in es:
            if e!=test:
                continue
            #print('\n', test, e, end=' ')
            print('')
            for embedding in embeddings:
                #print('\n', test, e,embedding,  end=' ')
                print(#ECT(embeddings[embedding], es[e][0], es[e][1], tests[test])[0],
                      #mod_ECT(embeddings[embedding], es[e][0], es[e][1], tests[test])[0],
                      #new1(embeddings[embedding], es[e][0], es[e][1], tests[test], K_MOD_ECT)[0]),
                      new2(embeddings[embedding], es[e][0], es[e][1], tests[test], EPSILON), 
                      end=' '
                )
                # print("here: new3",new3(embeddings[embedding], es[e][0], es[e][1], tests[test]))

if __name__=="__main__":
    
    print("Loading embeddings")
    emb_full = np.load('emb_all.npy', allow_pickle=True).item()
    for K_GENDER in [2, 3, 4, 8]:
        print('\n\n\n--------------', K_GENDER, '-------------------')
        run_tests([emb_full], K_GENDER=K_GENDER)
