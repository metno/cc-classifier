# cc-classifier
Trains a CNN for predicting cloud coverage based on images from webcams
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 128, 128, 128)     3584      
                                                                 
 batch_normalization (BatchN  (None, 128, 128, 128)    512       
 ormalization)                                                   
                                                                 
 conv2d_1 (Conv2D)           (None, 128, 128, 128)     147584    
                                                                 
 batch_normalization_1 (Batc  (None, 128, 128, 128)    512       
 hNormalization)                                                 
                                                                 
 max_pooling2d (MaxPooling2D  (None, 64, 64, 128)      0         
 )                                                               
                                                                 
 dropout (Dropout)           (None, 64, 64, 128)       0         
                                                                 
 conv2d_2 (Conv2D)           (None, 64, 64, 256)       295168    
                                                                 
 batch_normalization_2 (Batc  (None, 64, 64, 256)      1024      
 hNormalization)                                                 
                                                                 
 conv2d_3 (Conv2D)           (None, 64, 64, 256)       590080    
                                                                 
 batch_normalization_3 (Batc  (None, 64, 64, 256)      1024      
 hNormalization)                                                 
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 32, 32, 256)      0         
 2D)                                                             
                                                                 
 dropout_1 (Dropout)         (None, 32, 32, 256)       0         
                                                                 
 conv2d_4 (Conv2D)           (None, 32, 32, 512)       1180160   
                                                                 
 batch_normalization_4 (Batc  (None, 32, 32, 512)      2048      
 hNormalization)                                                 
                                                                 
 conv2d_5 (Conv2D)           (None, 32, 32, 512)       2359808   
                                                                 
 batch_normalization_5 (Batc  (None, 32, 32, 512)      2048      
 hNormalization)                                                 
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 16, 16, 512)      0         
 2D)                                                             
                                                                 
 dropout_2 (Dropout)         (None, 16, 16, 512)       0         
                                                                 
 conv2d_6 (Conv2D)           (None, 16, 16, 1024)      4719616   
                                                                 
 batch_normalization_6 (Batc  (None, 16, 16, 1024)     4096      
 hNormalization)                                                 
                                                                 
 conv2d_7 (Conv2D)           (None, 16, 16, 1024)      9438208   
                                                                 
 batch_normalization_7 (Batc  (None, 16, 16, 1024)     4096      
 hNormalization)                                                 
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 8, 8, 1024)       0         
 2D)                                                             
                                                                 
 dropout_3 (Dropout)         (None, 8, 8, 1024)        0         
                                                                 
 flatten (Flatten)           (None, 65536)             0         
                                                                 
 dense (Dense)               (None, 2048)              134219776 
                                                                 
 batch_normalization_8 (Batc  (None, 2048)             8192      
 hNormalization)                                                 
                                                                 
 dropout_4 (Dropout)         (None, 2048)              0         
                                                                 
 dense_1 (Dense)             (None, 9)                 18441     
                                                                 
=================================================================
Total params: 152,995,977
Trainable params: 152,984,201
Non-trainable params: 11,776
