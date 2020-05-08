IMPORT Python3 as Python;
IMPORT GNN.Tensor;
TensData := Tensor.R4.TensData;

/** ECL Image module
  * 
  * Overview:
  * This module tends towards the input, output and manipulation of images pertaining to neural network applications.  
  * This makes sure that the users of GNN do not spend time trying to preprocess the image database, as the functions in this module does it for them.
  * 
  * We can divide the module grossly as: -
  * 1) Input
  * The module takes image files such as jpg or bmp or png as input and converts it into raw image data using opencv functions in python. 
  * Not only image files, it takes well structured image datasets compressed as unsigned bytes, like the MNIST dataset by Yann LeCunn.  
  * 2) Output
  * The module takes raw byte data of the image from defined records and outputs the same as required. The user may use functions to output a dataset
  * of images as a PNG or JPG or they can output the images as a grid in the form of a PNG image for checking multiple images conveniently.  
  * 3) Conversions
  * This module handles the preprocessing. It can convert records containing images as byte data into Tensor data to be able to use for conversion 
  * into a tensor and train the neural network using the tensor.  
  * It also handles the conversion of Tensor data output from the network into images to visualise and see how the model performed. 
  */
EXPORT Image := MODULE

  /** This record stores the Image labels for testing models with images. A label is 1 byte unsigned integer.
    * Mostly used for MNIST dataset.
    * @field id Index of the label, so that index of image may be matched
    * @field label Label stored as a number to be used conveniently
    */
  SHARED IMG_labels := RECORD
    UNSIGNED id;
    UNSIGNED1 label;
  END;

  /** This record stores the images as per their index. Helps for maintaining the images and is simple to convert.
    * @field filename The name of the image as per the extracted dataset
    * @field image Image stored as an array of bytes which can be converted into integers or real values for using with neural networks.
    */  
  SHARED IMG_FORMAT := RECORD
    STRING filename;
    DATA image;
  END;

  /** This record stores the images as per their index. Helps for maintaining the images and is simple to convert.
    * @field id Index of the image which can also act as an identification factor for the image.
    * @field image Image stored as an array of bytes which can be converted into integers or real values for using with neural networks.
    */
  SHARED IMG_NUMERICAL := RECORD
    UNSIGNED8 id;
    DATA image;
    SET OF UNSIGNED imgDims;
  END;

  /** This function converts a compressed unsigned byte file, MNIST train images into a record for suitable input
    * It takes either a logical file directly given the path or a logical file sprayed as a BLOB. Bytes are manipulated to convert the file into images.  
    * @param filename A string which would hold the filename for the logical file name or the landing zone file.  
    * @return A dataset of images from the file in the form of IMG_NUMERICAL record
    */
  EXPORT DATASET(IMG_NUMERICAL) MNIST_train_images(STRING filename) := FUNCTION
    MNIST_FORMAT := RECORD
        DATA4 magic;
        DATA4 numImages;
        DATA4 numRows;
        DATA4 numCols;
        DATA47040000 contents;
    END;

    mnist_imgs := DATASET(filename, MNIST_FORMAT, FLAT);
    numRows := (>UNSIGNED1<)mnist_imgs[1].numRows[4];
    numCols := (>UNSIGNED1<)mnist_imgs[1].numCols[4];
    imgSize := numRows*numCols;
    numImages := (>UNSIGNED2<) (mnist_imgs[1].numImages[4] + mnist_imgs[1].numImages[3]);

    mnistOut := NORMALIZE(mnist_imgs[..numImages], numImages, TRANSFORM(IMG_NUMERICAL,
                                SELF.image := LEFT.contents[((COUNTER-1)*imgSize+1) .. (COUNTER*imgSize)],
                                SELF.id := COUNTER,
                                SELF.imgDims := [numRows, numCols, 1]));

    outRecs := DISTRIBUTE(mnistOut,id); 
    RETURN outRecs;                            
  END;

  /** This function converts a compressed unsigned byte file, MNIST test images into a record for suitable input
    * It takes either a logical file directly given the path or a logical file sprayed as a BLOB. Bytes are manipulated to convert the file into images.  
    * @param filename A string which would hold the filename for the logical file name or the landing zone file.  
    * @return A dataset of images from the file in the form of IMG_NUMERICAL record
    */
  EXPORT DATASET(IMG_NUMERICAL) MNIST_test_images(STRING filename) := FUNCTION
    MNIST_FORMAT := RECORD
        DATA4 magic;
        DATA4 numImages;
        DATA4 numRows;
        DATA4 numCols;
        DATA7840000 contents;
    END;

    mnist_imgs := DATASET(filename, MNIST_FORMAT, FLAT);
    numRows := (>UNSIGNED1<)mnist_imgs[1].numRows[4];
    numCols := (>UNSIGNED1<)mnist_imgs[1].numCols[4];
    imgSize := numRows*numCols;
    numImages := (>UNSIGNED2<) (mnist_imgs[1].numImages[4] + mnist_imgs[1].numImages[3]);

    mnistOut := NORMALIZE(mnist_imgs[..numImages], numImages, TRANSFORM(IMG_NUMERICAL,
                                SELF.image := LEFT.contents[((COUNTER-1)*imgSize+1) .. (COUNTER*imgSize)],
                                SELF.id := COUNTER,
                                SELF.imgDims := [numRows, numCols, 1]));

    outRecs := DISTRIBUTE(mnistOut,id); 
    RETURN outRecs;                            
  END;

  /** This function converts a compressed unsigned byte file, MNIST train labels into a record for suitable input
    * It takes either a logical file directly given the path or a logical file sprayed as a BLOB. Bytes are manipulated to convert the file into labels.  
    * @param filename A string which would hold the filename for the logical file name or the landing zone file.  
    * @return A dataset of labels from the file in the form of IMG_labels record
    */
  EXPORT DATASET(IMG_labels) MNIST_train_labels(STRING filename) := FUNCTION
    MNIST_FORMAT := RECORD
        DATA4 magic;
        DATA4 numImages;
        DATA60000 contents;
    END;

    mnist_lbls := DATASET(filename,MNIST_FORMAT,FLAT);
    numImages := (>UNSIGNED2<) (mnist_lbls[1].numImages[4] + mnist_lbls[1].numImages[3]);
    mnistOut := NORMALIZE(mnist_lbls[..numImages], numImages, TRANSFORM(IMG_labels, 
                                        SELF.label := (>UNSIGNED1<)LEFT.contents[COUNTER],
                                        SELF.id := COUNTER;));

    outRecs := DISTRIBUTE(mnistOut,id);
    RETURN outRecs;
  END;

  /** This function converts a compressed unsigned byte file, MNIST test labels into a record for suitable input
    * It takes either a logical file directly given the path or a logical file sprayed as a BLOB. Bytes are manipulated to convert the file into labels.  
    * @param filename A string which would hold the filename for the logical file name or the landing zone file.  
    * @return A dataset of labels from the file in the form of IMG_labels record
    */
  EXPORT DATASET(IMG_labels) MNIST_test_labels(STRING filename) := FUNCTION
    MNIST_FORMAT := RECORD
        DATA4 magic;
        DATA4 numImages;
        DATA10000 contents;
    END;

    mnist_lbls := DATASET(filename,MNIST_FORMAT,FLAT);
    numImages := (>UNSIGNED2<) (mnist_lbls[1].numImages[4] + mnist_lbls[1].numImages[3]);
    mnistOut := NORMALIZE(mnist_lbls[..numImages], numImages, TRANSFORM(IMG_labels, 
                                        SELF.label := (>UNSIGNED1<)LEFT.contents[COUNTER],
                                        SELF.id := COUNTER;));

    outRecs := DISTRIBUTE(mnistOut,id);
    RETURN outRecs;
  END;

  /** This function takes a logical file where multiple images may be sprayed as a blob with the prefix:[filename,filesize].
    * After obtaining dataset, converts into a dataset which numbers as indexes so as to traverse easier and making tensor
    * @param filename String containing the logical file of the image dataset
    * @return IMG_NUMERICAL dataset which can be converted to Tensor easily using a conversion function 
    */
  EXPORT DATASET(IMG_NUMERICAL) GetImages(STRING filename, SET OF UNSIGNED dims, BOOLEAN resize = TRUE) := FUNCTION
    DATA ReadImage(DATA image, SET OF UNSIGNED dims, BOOLEAN color, BOOLEAN resize) := EMBED(Python)
      import cv2
      import numpy as np
      image_np = np.frombuffer(image, dtype='uint8')
      if color:
        img = cv2.imdecode(image_np,cv2.IMREAD_COLOR)
      else:
        img = cv2.imdecode(image_np,cv2.IMREAD_GRAYSCALE)  
      dims = tuple(dims)
      if resize:
        img = cv2.resize(img, dims)
      else:
        y,x,_ = img.shape
        h,k = dims
        img = img[int((y-k)/2):int((y+k)/2), int((x-h)/2):int((x+h)/2)]
      return bytearray(img)
    ENDEMBED;

    SET OF INTEGER GetImageDimensions(DATA image) := EMBED(Python)
        import cv2
        import numpy as np 

        nparr = np.frombuffer(bytearray(image), dtype='uint8')
        img_np = cv2.imdecode(nparr,cv2.IMREAD_UNCHANGED)
        if len(img_np.shape) == 2:
          dims = list(img_np.shape)
          dims.append(1)
        else:
          dims = list(img_np.shape)  
        return dims;
    ENDEMBED;

    imageData := DATASET(filename, IMG_FORMAT, FLAT);
    imgDims := dims + [GetImageDimensions(imageData[1].image)[3]];
    color := IF(imgDims[3] = 3, TRUE, FALSE);

    imageNumerical := PROJECT(imageData, TRANSFORM(IMG_NUMERICAL,
                                          SELF.id := COUNTER,
                                          SELF.image := ReadImage(LEFT.image, dims, color,resize),
                                          SELF.imgDims := imgDims
                                          ));
    return imageNumerical;
  END;

  /** This function converts an IMG_Numerical record format to TensData format so that it can be passed into makeTensor
    * This makes it much easier to give image datasets as input to neural networks
    * This is different because the files can be in various formats, or even in RGB or HSV formats to work on
    * @param imgDataset The dataset of images which need to be converted to Tensor Data format
    * @return Tensor data of the images so as to send to the makeTensor with ease
    */
  EXPORT DATASET(TensData) ImgtoTens(DATASET(IMG_NUMERICAL) imgDataset) := FUNCTION
    imgShape := imgDataset[1].imgDims;

    imgRows := imgShape[1];
    imgCols := imgShape[2];
    imgChannels := imgShape[3];

    imgSize := imgRows * imgCols * imgChannels;

    tens := NORMALIZE(imgDataset, imgSize, TRANSFORM(TensData,
                        SELF.indexes := [LEFT.id, (COUNTER-1) DIV imgCols+1, (COUNTER-1) % imgCols +1, (COUNTER-1) % imgChannels + 1],
                        SELF.value := ( (REAL) (>UNSIGNED1<) LEFT.image[counter] )/127.5 - 1 ));
    RETURN tens;                    
  END;

  /** This function converts the Tensor data output from the neural network into an Image dataset so it can be output in various forms
    * This is done so that the user can visualise and understand how successful their model has been. It converts a set of tensor values to bytes to achieve this.  
    * @param tens Tensor data that is needed to be converted to visualise as an image
    * @return An image dataset which can be output in any way as it's a simple byte array
    */
  EXPORT DATASET(IMG_NUMERICAL) TenstoImg(DATASET(TensData) tens) := FUNCTION
    DATA giveBytes(SET OF UNSIGNED input) := EMBED(Python)
        return bytearray(input)
    ENDEMBED;

    numImages := MAX(tens, tens.indexes[1]);
    numRows := MAX(tens, tens.indexes[2]);
    numCols := MAX(tens, tens.indexes[3]);
    numChannels := MAX(tens, tens.indexes[4]);
    imageDataset := DATASET(numImages,TRANSFORM(IMG_NUMERICAL,
                        SELF.id := COUNTER,
                        SELF.image := giveBytes(SET(tens(indexes[1]=COUNTER),(UNSIGNED)((value+1)*127.5))),
                        SELF.imgDims := [numRows, numCols, numChannels]));
    RETURN imageDataset;                    
  END;

  /** This function returns the image dataset which when desprayed with the prefix:[filename,filesize] will provide the JPG images of the files
    * It has a flaw which is that it can work only on MNIST for now.  
    * @param mnist Take the image dataset of mnist images to convert to JPG images
    * @return Image dataset having encoded JPG in bytearray of Images
    */
  EXPORT DATASET(IMG_FORMAT) OutputasJPG(DATASET(IMG_NUMERICAL) mnist, STRING filename = '') := FUNCTION
    DATA makeJPG(DATA image, SET OF UNSIGNED dims) := EMBED(Python)
        import numpy as np
        import cv2

        dims = dims[:2]
        image_np = np.frombuffer(image, dtype=np.uint8)
        image_mat = image_np.reshape(dims)
        img_encode = cv2.imencode('.jpg', image_mat)[1]
        return bytearray(img_encode)
    ENDEMBED;

    mnist_jpg := PROJECT(mnist, TRANSFORM(IMG_FORMAT,
                        SELF.filename := LEFT.id + filename + '.jpg';
                        SELF.image := makeJPG(LEFT.image, LEFT.imgDims);
                        ));
    return mnist_jpg;                    
  END;

  /** This function returns the image dataset which when desprayed with the prefix:[filename,filesize] will provide the PNG images of the files
    * It has a flaw which is that it can work only on MNIST for now.  
    * @param mnist Take the image dataset of mnist images to convert to PNG images
    * @return Image dataset having encoded PNG in bytearray of Images
    */
  EXPORT DATASET(IMG_FORMAT) OutputasPNG(DATASET(IMG_NUMERICAL) mnist, STRING filename = '') := FUNCTION
    DATA makePNG(DATA image, SET OF UNSIGNED dims) := EMBED(Python)
        import numpy as np
        import cv2

        dims = dims[:2]
        image_np = np.frombuffer(image, dtype=np.uint8)
        image_mat = image_np.reshape(dims)
        img_encode = cv2.imencode('.png', image_mat)[1]
        return bytearray(img_encode)
    ENDEMBED;

    mnist_png := PROJECT(mnist, TRANSFORM(IMG_FORMAT,
                        SELF.filename := LEFT.id + filename + '_mnist.png';
                        SELF.image := makePNG(LEFT.image, LEFT.imgDims);
                        ));
    return mnist_png;                    
  END;

  /** This function converts mnist image dataset to an a grid of (r,c) where r*c should be size of the dataset.  
    * It makes a grid of (r,c) and encodes the image into a PNG to output. This helps for better understanding of the data which is obtained as output.  
    * @param mnist Dataset of mnist images as usual 
    * @param r Number of rows in the grid
    * @param c Number of columns in the grid
    * @param epochnum Number of epochs the dataset was trained for. This is only used for name of the image.
    * @return Dataset of 1 image which contains a PNG file with the resultant images in a grid
    */
  EXPORT DATASET(IMG_FORMAT) OutputGrid(DATASET(IMG_NUMERICAL) mnist, INTEGER r, INTEGER c, STRING filename) := FUNCTION
    DATA makeGrid(SET OF DATA images, Integer r, Integer c, SET OF UNSIGNED dims) := EMBED(Python)
        import matplotlib.pyplot as plt
        import numpy as np
        import cv2

        numRows,numCols,_ = dims
        dims = (numRows, numCols)
        fig, axs = plt.subplots(r, c)
        cnt = 0
        if r == 1 and c == 1:
          assert 1==0, 'Error: Grid of (1,1) not possible. Use OutputasPNG function instead!'
        elif r == 1:
          for i in range(c):
            image = images[cnt]
            image_np = np.frombuffer(image, dtype=np.uint8)
            image_mat = image_np.reshape(dims)
            axs[i].imshow(image_mat[:,:], cmap='gray')
            axs[i].axis('off')
            cnt += 1
        elif c == 1:
          for i in range(r):
            image = images[cnt]
            image_np = np.frombuffer(image, dtype=np.uint8)
            image_mat = image_np.reshape(dims)
            axs[i].imshow(image_mat[:,:], cmap='gray')
            axs[i].axis('off')
            cnt += 1
        else:        
          for i in range(r):
            for j in range(c):
              image = images[cnt]
              image_np = np.frombuffer(image, dtype=np.uint8)
              image_mat = image_np.reshape(dims)
              axs[i,j].imshow(image_mat[:,:], cmap='gray')
              axs[i,j].axis('off')
              cnt += 1
        fig.canvas.draw()        
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))  
        plt.close()  
        img_encode = cv2.imencode('.png', image_from_plot)[1]
        return bytearray(img_encode)
    ENDEMBED;

    mnist_grid := DATASET(1, TRANSFORM(IMG_FORMAT,
                        SELF.filename := filename + '.png',
                        SELF.image := makeGrid(SET(mnist, image), r, c, mnist[1].ImgDims)
                        ));
    return mnist_grid;    
  END;
END; 