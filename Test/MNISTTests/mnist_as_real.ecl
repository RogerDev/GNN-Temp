/*
 Helper file to load the MNIST dataset for use in GNN.

 To get training data: mnist_as_real.train;
 To get testing data: mnist_as_real.test;
*/
ImageType := DATA784;  

mnist_dt := RECORD
      INTEGER1  label;
      ImageType image;
END;

mnist_dt_set := RECORD
			UNSIGNED8 id;
      UNSIGNED1 label;
      SET OF UNSIGNED1 pixel;
END;


GetSet(ImageType I) := FUNCTION
  PixRec := {UNSIGNED1 Pixels};
  PixDS  := DATASET(SIZEOF(I),
                    TRANSFORM(PixRec,
                              SELF.Pixels := (>UNSIGNED1<)I[COUNTER]));
  RETURN SET(PixDS,Pixels);
END;

GetSetREAL(ImageType I) := FUNCTION
  PixRec := {REAL4 Pixels};
  PixDS  := DATASET(SIZEOF(I),
                    TRANSFORM(PixRec,
                              SELF.Pixels := (>UNSIGNED1<)I[COUNTER]));
  RETURN SET(PixDS,Pixels);
END;


train0 := DATASET('~mnist::train', mnist_dt, THOR); 


test0 := DATASET('~mnist::test', mnist_dt,THOR);

trainDat := PROJECT(train0,
             TRANSFORM(mnist_dt_set,
											 SELF.id := COUNTER,
                       SELF.label := (UNSIGNED1)LEFT.label,
                       SELF.pixel := GetSet(LEFT.image)));
											 			 
testDat:= PROJECT(test0,
					TRANSFORM(mnist_dt_set,
										SELF.id := COUNTER,
										SELF.label := (UNSIGNED1)LEFT.label,
										SELF.pixel := GetSet(LEFT.image)));



trainDatREAL := PROJECT(train0,
             TRANSFORM(mnist_dt_set,
											 SELF.id := COUNTER,
                       SELF.label := (REAL4)LEFT.label,
                       SELF.pixel := GetSetREAL(LEFT.image)));
testDatREAL := PROJECT(test0,
             TRANSFORM(mnist_dt_set,
											 SELF.id := COUNTER,
                       SELF.label := (REAL4)LEFT.label,
                       SELF.pixel := GetSetREAL(LEFT.image)));

//OUTPUT(trainDatREAL);
//OUTPUT(testDatREAL);

EXPORT mnist_as_real := MODULE
		EXPORT train := trainDatREAL;//trainDat;
		EXPORT test := testDatREAL;//testDat;
END;


