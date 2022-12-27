from PIL import Image
import numpy as np



def convolve(matrix,kernal,padding,stride):
    #get shape
    x_KernalShape = kernal.shape[0]
    y_kernalShape = kernal.shape[1]
    x_ImageShape = matrix.shape[0]
    y_ImageShape = matrix.shape[1]

    #get shape of convolving result: (I - k + 1)x(I - k +1) when padding=0,strides=1
    xresult = int(((x_ImageShape - x_KernalShape + 2 *padding) / stride) + 1)
    yresult = int(((y_ImageShape - y_kernalShape + 2 *padding) / stride) + 1)
    result = np.zeros((xresult,yresult))

    #Iterate through the image
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i,j] = (matrix[i:i + x_KernalShape , j:j + y_kernalShape] * kernal).sum()
    return result


if __name__ == '__main__':
    #**********************
    #for i(a)
    #return a random integer from 1 to 8
    matrix = np.random.randint(1,8,size=(6,6))
    kernel = np.random.randint(1,8,size=(3,3))
    convolve_result = convolve(matrix,kernel,0,1)
    #print(convolve_result)

    #*********************
    #for i(b)
    im = Image.open('sun.jpg')
    rgb = np.array(im.convert('RGB'))
    r = rgb[:,:,0]
    #Image.fromarray(np.uint(r)).show()

    kernel1 = np.array([[-1,-1,1],[-1,8,-1],[-1,-1,-1]])
    convolve_result1 = convolve(r,kernel1,0,1)
    #Image.fromarray(np.uint8(convolve_result1)).show()

    kernel2 = np.array([[0,-1,0],[-1,8,-1],[0,-1,0]])
    convolve_result2 = convolve(r,kernel2,0,1)
    Image.fromarray(np.uint8(convolve_result2)).show()

    #*********************
    #for
