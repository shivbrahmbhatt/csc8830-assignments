import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Using the synthetic setup to match your test
    image = np.zeros((300, 300), dtype=np.float32)
    cv2.rectangle(image, (50, 50), (250, 250), 255, -1)
    cv2.circle(image, (150, 150), 50, 0, -1)

    # Kernel
    ksize = 15
    sigma = 3
    k_1d = cv2.getGaussianKernel(ksize, sigma)
    kernel = np.outer(k_1d, k_1d)

    # SPATIAL FILTERING
    # anchor=(-1,-1) tells OpenCV to align the kernel center (7,7) with the pixel.
    spatial_result = cv2.filter2D(image, -1, kernel, 
                                  anchor=(-1, -1), 
                                  borderType=cv2.BORDER_CONSTANT)

    # FREQUENCY DOMAIN FILTERING 
    # padded kernel matching the image size
    kernel_padded = np.zeros_like(image)
    
    # Calculate the center offset of the kernel
    kh, kw = kernel.shape
    center_y, center_x = kh // 2, kw // 2
    
    # 15x15 kernel in the top-left (0,0) first, then shift it.
    kernel_padded[:kh, :kw] = kernel
    
    kernel_padded = np.roll(kernel_padded, -center_y, axis=0)
    kernel_padded = np.roll(kernel_padded, -center_x, axis=1)

    # Compute FFT
    dft_image = np.fft.fft2(image)
    dft_kernel = np.fft.fft2(kernel_padded)
    
    # Convolve (Multiply in Frequency Domain)
    dft_result = dft_image * dft_kernel
    
    # Inverse FFT
    freq_result = np.real(np.fft.ifft2(dft_result))

    # COMPARISON
    # Calculate absolute difference
    diff = np.abs(spatial_result - freq_result)
    
    print(f"Mean Difference: {np.mean(diff):.5f}")
    print(f"Max Difference:  {np.max(diff):.5f}")

    # VISUALIZATION
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.title("Spatial (filter2D)")
    plt.imshow(spatial_result, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Frequency (FFT)")
    plt.imshow(freq_result, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f"Difference (Max: {np.max(diff):.2f})")
    plt.imshow(diff, cmap='jet')
    plt.colorbar()
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
