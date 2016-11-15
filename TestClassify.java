package examples;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import parsers.NewRDPParserFileLine;
import utils.ConfigReader;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class TestClassify
{
	/*
	public synchronized static void main(String[] args) throws Exception
	{
		Random random = new Random(0);
		protected int numPermutations = 25;
		protected int m_numExecutionSlots = 4;

		for( int x=1 ; x < NewRDPParserFileLine.TAXA_ARRAY.length; x++)
		{
			
			System.out.println(NewRDPParserFileLine.TAXA_ARRAY[x]);
			BufferedWriter writer = new BufferedWriter(new FileWriter(new File(ConfigReader.getAdenomasWekaDir() + File.separator + NewRDPParserFileLine.TAXA_ARRAY[x] +"_Adenomas.txt" )));
			
			writer.write("ad1\tad2\tcross\n");
			
			File adenomas = new File("C:\\adenomasRelease\\spreadsheets\\pivoted_" + NewRDPParserFileLine.TAXA_ARRAY[x] + 	"LogNormalWithMetadataBigSpace.arff");
			List<Double> firstARoc = getPercentCorrectForOneFile(adenomas, numPermutations,random);
			
			File ad2 = new File("C:\\tope_Sep_2015\\spreadsheets\\" + NewRDPParserFileLine.TAXA_ARRAY[x] + "asColumnsLogNormalPlusMetadataBigSpace.arff");
			
			List<Double> secondROC = getPercentCorrectForOneFile(ad2, numPermutations,random);;
			
			Instances trainData = DataSource.read(ad2.getAbsolutePath());
			Instances testData = DataSource.read(adenomas.getAbsolutePath());
			trainData.setClassIndex(trainData.numAttributes() -1);
			testData.setClassIndex(testData.numAttributes() -1);
			
			List<Double> crossROC =	getRocForTrainingToTest(trainData, testData, random, numPermutations);
			
			for(int y=0;y < firstARoc.size(); y++)
				writer.write(firstARoc.get(y) + "\t" + secondROC.get(y) + "\t" + crossROC.get(y) + "\n");
			
			writer.flush();  writer.close();
		}
		
	}
	*/
	
	public synchronized static void main(String[] args) throws Exception
	{
		Random random = new Random(0);
		int numPermutations = 25;

		for( int x=1 ; x < NewRDPParserFileLine.TAXA_ARRAY.length; x++)
		{
			BufferedWriter writer = new BufferedWriter(new FileWriter(new File(
					ConfigReader.getAdenomasWekaDir() + File.separator + 
						"adenomasShuffled" + NewRDPParserFileLine.TAXA_ARRAY[x] + ".txt")));
			
			writer.write("type\tvalue\n");
			
			System.out.println(NewRDPParserFileLine.TAXA_ARRAY[x]);
			File adenomas = new File("C:\\adenomasRelease\\spreadsheets\\pivoted_" + 
					NewRDPParserFileLine.TAXA_ARRAY[x] + 	"LogNormalWithMetadataBigSpace.arff");

			File ad2 = new File("C:\\tope_Sep_2015\\spreadsheets\\" + 
					NewRDPParserFileLine.TAXA_ARRAY[x] + "asColumnsLogNormalPlusMetadataBigSpace.arff");
			
			Instances testData = DataSource.read(adenomas.getAbsolutePath());

			List<Double> percentCorrect =
			getPercentCorrectFromScrambles(ad2, testData, random, 1,false);
			
			writer.write("true\t" + percentCorrect.get(0) + "\n");
			
			percentCorrect =
					getPercentCorrectFromScrambles(ad2, testData, random, numPermutations,true);
			
			for( Double d : percentCorrect)
				writer.write("shuffle\t" + d + "\n");
					
			writer.flush(); writer.close();
			
		}
	}

	
	private synchronized static void scrambeLastColumn( Instances instances, Random random )
	{
		List<Double> list = new ArrayList<Double>();
		
		for(Instance i : instances)
			list.add(i.value(i.numAttributes()-1));
		
		Collections.shuffle(list,random);
		
		for( int x=0; x < instances.size(); x++)
		{
			Instance i = instances.get(x);
			i.setValue(i.numAttributes()-1, list.get(x));
		}
	}
	 
	 	public void More ...setNumExecutionSlots(int numSlots) 
	 	{
    		m_numExecutionSlots = numSlots;
		 }
	 
	 	 public int More ...getNumExecutionSlots() 
	  	{
			return m_numExecutionSlots;
 		}
 		
 		public String More ...numExecutionSlotsTipText() 
	 	{
			return "The number of execution slots (threads) to use for " +
			"constructing the ensemble.";
		}
	 
	public static List<Double> getPercentCorrectFromScrambles( File trainingDataFile, 
				Instances testData, Random random, int numPermutations,
							boolean scramble) throws Exception
	{
		System.out.println("In scramble");
		List<Double> aList = new ArrayList<Double>();
		Instances trainData = DataSource.read(trainingDataFile.getAbsolutePath());
		
		for( int x=0; x < numPermutations; x++)
		{
			if( scramble)
				scrambeLastColumn(trainData, random);
			
			trainData.setClassIndex(trainData.numAttributes() -1);
			testData.setClassIndex(testData.numAttributes() -1);
			
			AbstractClassifier rf = new RandomForest();
			rf.buildClassifier(trainData);
			Evaluation ev = new Evaluation(trainData);
			ev.evaluateModel(rf, testData);
			if( x % 20 ==0)
			System.out.println("cross " + x + " " + ev.areaUnderROC(0) + " " + ev.pctCorrect());
			aList.add(ev.pctCorrect());
		}
		
		return aList;
	}
	
	public synchronized static List<Double> getRocForTrainingToTest(Instances trainingData, Instances testData,
				Random random, int numIterations) throws Exception
	{

		List<Double> rocAreas = new ArrayList<Double>();
		
		for( int x=0; x < numIterations; x++)
		{

			Instances halfTrain = new Instances(trainingData, trainingData.size() / 2 );
			
			for(Instance i : trainingData)
				if( random.nextFloat() <= 0.5)
					halfTrain.add(i);
			
			System.out.println(halfTrain.size() + " " + trainingData.size());
			
			AbstractClassifier rf = new RandomForest();
			rf.buildClassifier(halfTrain);
			Evaluation ev = new Evaluation(halfTrain);
			ev.evaluateModel(rf, testData);
			System.out.println("cross " + x + " " + ev.areaUnderROC(0) + " " + ev.pctCorrect());
			rocAreas.add(ev.pctCorrect());
		}
		
		return rocAreas;
	}
	
	public static synchronized List<Double> getPercentCorrectForOneFile( File inFile, int numPermutations, Random random ) 
				throws Exception
	{
		List<Double> percentCorrect = new ArrayList<Double>();
		
		for( int x=0; x< numPermutations; x++)
		{
			Instances data = DataSource.read(inFile.getAbsolutePath());
			data.setClassIndex(data.numAttributes() -1);
			Evaluation ev = new Evaluation(data);
			AbstractClassifier rf = new RandomForest();
			
			//rf.buildClassifier(data);
			ev.crossValidateModel(rf, data, 25, random);
			//System.out.println(ev.toSummaryString("\nResults\n\n", false));
			//System.out.println(x + " " + ev.areaUnderROC(0) + " " + ev.pctCorrect());
			percentCorrect.add(ev.pctCorrect());
		}
		
		return percentCorrect;
		
	}
}
