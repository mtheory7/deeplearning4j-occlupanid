package com.mtheory7.utils;

import org.bytedeco.javacv.OpenCVFrameConverter; // Added missing import
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.datavec.image.data.ImageWritable;
import org.datavec.image.transform.BaseImageTransform;

import java.util.Random;

import static org.bytedeco.opencv.global.opencv_core.BORDER_CONSTANT;
import static org.bytedeco.opencv.global.opencv_core.copyMakeBorder;

public class PadToSquare extends BaseImageTransform<Mat> {

    // 1. Create a strongly-typed converter specifically for Mat objects
    private static final OpenCVFrameConverter.ToMat toMatConverter = new OpenCVFrameConverter.ToMat();

    public PadToSquare() {
        super(null);
    }

    @Override
    protected ImageWritable doTransform(ImageWritable image, Random random) {
        if (image == null) {
            return null;
        }

        // 2. Use our strongly-typed converter instead of the inherited 'converter'
        Mat mat = toMatConverter.convert(image.getFrame());
        int h = mat.rows();
        int w = mat.cols();

        if (h == w) {
            return image;
        }

        int diff = Math.abs(h - w);
        int top = 0, bottom = 0, left = 0, right = 0;

        if (h > w) {
            left = diff / 2;
            right = diff - left;
        } else {
            top = diff / 2;
            bottom = diff - top;
        }

        Mat paddedMat = new Mat();

        copyMakeBorder(mat, paddedMat, top, bottom, left, right, BORDER_CONSTANT, new Scalar(0, 0, 0, 0));

        // 3. Convert the padded Mat back to an ImageWritable
        return new ImageWritable(toMatConverter.convert(paddedMat));
    }
}