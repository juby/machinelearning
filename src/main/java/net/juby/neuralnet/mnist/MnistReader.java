package net.juby.neuralnet.mnist;

import net.juby.exceptions.MalformedInputDataException;

import static java.lang.String.format;

import java.io.ByteArrayOutputStream;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

/**
 * Provides tools for reading data from the
 * <a href="http://yann.lecun.com/exdb/mnist/">MNIST database</a>.
 * Code originally by
 * <a href="https://github.com/jeffgriffith/mnist-reader">Jeff Griffith</a>.
 */
public class MnistReader {
	private static final int LABEL_FILE_MAGIC_NUMBER = 2049;
	private static final int IMAGE_FILE_MAGIC_NUMBER = 2051;

	/**
	 * Extracts the desired outputs for test or training cases from the MNIST data.
	 * @param infile the file location of the MNIST data
	 * @return an array of the desired outputs
	 */
	public static int[] getLabels(String infile) {

		ByteBuffer bb = loadFileToByteBuffer(infile);

		assertMagicNumber(LABEL_FILE_MAGIC_NUMBER, bb.getInt());

		int numLabels = bb.getInt();
		int[] labels = new int[numLabels];

		for (int i = 0; i < numLabels; ++i)
			labels[i] = bb.get() & 0xFF; // To unsigned

		return labels;
	}

	/**
	 * Extracts the pixel data for test or training cases from the MNIST data.
	 * @param infile the file location of the MNIST data
	 * @return a List of two dimensional integer arrays holding the pixel values
	 */
	public static List<int[][]> getImages(String infile) {
		ByteBuffer bb = loadFileToByteBuffer(infile);

		assertMagicNumber(IMAGE_FILE_MAGIC_NUMBER, bb.getInt());

		int numImages = bb.getInt();
		int numRows = bb.getInt();
		int numColumns = bb.getInt();
		List<int[][]> images = new ArrayList<>();

		for (int i = 0; i < numImages; i++)
			images.add(readImage(numRows, numColumns, bb));

		return images;
	}

	/**
	 * Extracts the pixel information for a single image.
	 * @param numRows the number of rows in the image
	 * @param numCols the number of columns in the image
	 * @param bb the raw data for an image
	 * @return a two dimensional array holding the pixel information for the image
	 */
	private static int[][] readImage(int numRows, int numCols, ByteBuffer bb) {
		int[][] image = new int[numRows][];
		for (int row = 0; row < numRows; row++)
			image[row] = readRow(numCols, bb);
		return image;
	}

	/**
	 * Extracts the pixel information for a single row of an image.
	 * @param numCols the number of pixels in the row
	 * @param bb the raw data for the row
	 * @return an array holding the pixel information for the row
	 */
	private static int[] readRow(int numCols, ByteBuffer bb) {
		int[] row = new int[numCols];
		for (int col = 0; col < numCols; ++col)
			row[col] = bb.get() & 0xFF; // To unsigned
		return row;
	}

	/**
	 * A sanity check to ensure that the MNIST file used is the correct data.
	 * @param expectedMagicNumber the identifier for the file intended to be used
	 * @param magicNumber the identifier extracted from the MNIST file that is used
	 */
	private static void assertMagicNumber(int expectedMagicNumber, int magicNumber) {
		if (expectedMagicNumber != magicNumber) {
			switch (expectedMagicNumber) {
			case LABEL_FILE_MAGIC_NUMBER:
				throw new RuntimeException("This is not a label file.");
			case IMAGE_FILE_MAGIC_NUMBER:
				throw new RuntimeException("This is not an image file.");
			default:
				throw new MalformedInputDataException(
						format("Expected magic number %d, found %d", expectedMagicNumber, magicNumber));
			}
		}
	}

	/*
	 * Just very ugly utilities below here. Best not to subject yourself to
	 * them. ;-)
	 */

	/**
	 * Converts the MNIST file into a format more easily used by the tools.
	 * @param infile the file location of the MNIST data
	 * @return the converted MNIST data
	 */
	private static ByteBuffer loadFileToByteBuffer(String infile) {
		return ByteBuffer.wrap(loadFile(infile));
	}

	/**
	 * Converts an MNIST file into an array of bytes.
	 * @param infile the file location of the MNIST data
	 * @return the converted MNIST data
	 */
	private static byte[] loadFile(String infile) {
		try {
			RandomAccessFile f = new RandomAccessFile(infile, "r");
			FileChannel chan = f.getChannel();
			long fileSize = chan.size();
			ByteBuffer bb = ByteBuffer.allocate((int) fileSize);
			chan.read(bb);
			bb.flip();
			ByteArrayOutputStream baos = new ByteArrayOutputStream();
			for (int i = 0; i < fileSize; i++)
				baos.write(bb.get());
			chan.close();
			f.close();
			return baos.toByteArray();
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	/**
	 * Generates ASCII art of an MNIST image.
	 * @param image a two dimensional array of pixel values for an MNIST image
	 * @return the ASCII art of the image
	 */
	private static String renderImage(int[][] image) {
		StringBuilder sb = new StringBuilder();

		for (int[] ints : image) {
			sb.append("|");
			for (int pixelVal : ints) {
				if (pixelVal == 0)
					sb.append(" ");
				else if (pixelVal < 256 / 3)
					sb.append(".");
				else if (pixelVal < 2 * (256 / 3))
					sb.append("x");
				else
					sb.append("X");
			}
			sb.append("|\n");
		}

		return sb.toString();
	}
}
